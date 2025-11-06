from typing import Any, Dict, Iterable, List, Sequence, Tuple
import numpy as np
import pandas as pd
import math
from scipy.stats import kendalltau, rankdata
from openbox.utils.history import History, Observation
from openbox import logger


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denom = weights.sum()
    if denom <= 0:
        return 0.0
    return float(np.sum(weights * values) / denom)


def _weighted_cov(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    mean_x = _weighted_mean(x, weights)
    mean_y = _weighted_mean(y, weights)
    centered_prod = (x - mean_x) * (y - mean_y)
    denom = weights.sum()
    if denom <= 0:
        return 0.0
    return float(np.sum(weights * centered_prod) / denom)


def _weighted_corr(x: np.ndarray, y: np.ndarray, weights: np.ndarray, method: str) -> float:
    if len(x) < 3 or len(y) < 3:
        return 0.0

    if method == "spearman":
        x = rankdata(x)
        y = rankdata(y)
    elif method == "kendall":
        # Weighted Kendall's tau is non-trivial; fall back to standard tau on the
        # filtered subset for now.
        tau, _ = kendalltau(x, y)
        return float(tau) if not np.isnan(tau) else 0.0
    elif method != "pearson":
        raise ValueError(f"Unknown correlation method: {method}")

    cov = _weighted_cov(x, y, weights)
    var_x = _weighted_cov(x, x, weights)
    var_y = _weighted_cov(y, y, weights)
    denom = math.sqrt(var_x * var_y) if var_x > 0 and var_y > 0 else 0.0
    if denom <= 0:
        return 0.0
    corr = cov / denom
    return float(corr)


def _safe_weighted_corr(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    method: str,
) -> float:
    mask = np.isfinite(x) & np.isfinite(y) & (weights > 0)
    if mask.sum() < 3:
        return 0.0
    return _weighted_corr(x[mask], y[mask], weights[mask], method)


# ---------------------------------------------------------------------------
# Observation, Record, History and DataFrame utilities
# ---------------------------------------------------------------------------
def config_to_dict(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, "get_dictionary"):
        try:
            return dict(config.get_dictionary())
        except Exception:
            pass
    try:
        return dict(config)
    except Exception:
        return {}


def _observation_to_record(observation: Observation, sql_type: str) -> Dict[str, float]:
    """Convert an Observation into a flat record for tabular analysis.
    
    Args:
        - observation: Observation object, which contains:
            - extra_info: Dict[str, Any]
                - origin: str
                - qt_time: Dict[str, float] (query name -> time)
                - et_time: Dict[str, float] (query name -> time)
            - objectives: List[float]
            - elapsed_time: float
        - sql_type: SQL type ('qt' or 'et')
    
    Returns:
        - Dictionary with SQL times and other metrics:
            - objective: float
            - elapsed_time: float
            - {sql_type}_{query_name}: float, for each query in sql_times
    """

    extra_info = getattr(observation, "extra_info", None) or {}
    sql_times: Dict[str, float] = extra_info.get(f"{sql_type}_time", {})

    record: Dict[str, float] = {}

    objectives = getattr(observation, "objectives", None)
    if objectives:
        record["objective"] = float(objectives[0])

    for sql_name, value in sql_times.items():
        column_name = f"{sql_type}_{sql_name}"
        record[column_name] = float(value)

    elapsed_time = getattr(observation, "elapsed_time", None)
    record["elapsed_time"] = float(elapsed_time) if elapsed_time is not None and np.isfinite(elapsed_time) else float("inf")

    return record


def _compute_calibration_factor(
    history: History,
    reference_history: History,
    sql_type: str = "qt",
) -> float:
    """
    Compute calibration factor based on the first configuration.
    
    Since all histories share the same first configuration, we can use it to
    calibrate time differences between different task runs.
    
    Returns:
        Calibration factor (reference_time / history_time), or 1.0 if not available
    """
    if len(history) == 0 or len(reference_history) == 0:
        return 1.0
    
    ref_obs: Observation = reference_history.observations[0]
    obs: Observation = history.observations[0]
    
    ref_record = _observation_to_record(ref_obs, sql_type=sql_type)
    record = _observation_to_record(obs, sql_type=sql_type)
    
    ref_objective = ref_record.get("objective", float("inf"))
    objective = record.get("objective", float("inf"))
    
    if not np.isfinite(ref_objective) or not np.isfinite(objective) or objective <= 0:
        return 1.0
    
    return float(ref_objective / objective)


def _aggregate_history_records(
    history: History,
    sql_type: str = "qt",
    top_ratio: float = 1.0,
    calibration_factor: float = 1.0,
) -> Dict[str, float]:
    """
    Aggregate observations within a single history by:
    1. Filtering to top_ratio of observations (sorted by objective)
    2. Averaging qt/et times for filtered observations
    3. Applying calibration factor
    
    Args:
        history: History object containing observations
        sql_type: SQL type ('qt' or 'et')
        top_ratio: Ratio of top observations to keep (0.0-1.0)
        calibration_factor: Factor to calibrate time differences
        
    Returns:
        Aggregated record dictionary with averaged SQL times
    """
    if len(history) == 0:
        return {}
    
    observations: List[Observation] = list(history.observations)
    valid_obs = [
        obs for obs in observations
        if hasattr(obs, "objectives") and obs.objectives and np.isfinite(obs.objectives[0])
    ]
    valid_obs.sort(key=lambda obs: obs.objectives[0] if obs.objectives else float("inf"))

    top_count = max(1, int(len(valid_obs) * top_ratio))
    filtered_obs = valid_obs[: top_count]

    sql_prefix = f"{sql_type}_"
    sql_times_dict: Dict[str, List[float]] = {}
    spark_times: List[float] = []
    objectives: List[float] = []
    elapsed_times: List[float] = []
    
    for obs in filtered_obs:
        record = _observation_to_record(obs, sql_type=sql_type)

        for key, value in record.items():
            if key.startswith(sql_prefix):
                sql_name = key[len(sql_prefix): ]   # remove sql_prefix from key
                if sql_name not in sql_times_dict:
                    sql_times_dict[sql_name] = []
                if np.isfinite(value):
                    sql_times_dict[sql_name].append(float(value) * calibration_factor)
        
        if "objective" in record and np.isfinite(record["objective"]):
            objectives.append(float(record["objective"]) * calibration_factor)
        if "elapsed_time" in record and np.isfinite(record["elapsed_time"]):
            elapsed_times.append(float(record["elapsed_time"]) * calibration_factor)
    
    aggregated: Dict[str, float] = {}
    for sql_name, times in sql_times_dict.items():
        aggregated[f"{sql_prefix}{sql_name}"] = float(np.mean(times)) if times else float("inf")
    aggregated["objective"] = float(np.mean(objectives)) if objectives else float("inf")
    aggregated["elapsed_time"] = float(np.mean(elapsed_times)) if elapsed_times else aggregated["objective"]

    return aggregated


def build_weighted_dataframe(
    histories_with_weights: Sequence[Tuple[History, float]],
    sql_type: str = "qt",
    top_ratio: float = 1.0,
) -> pd.DataFrame:
    """
    Build weighted dataframe from multiple histories.
    1. Filters each history to top_ratio observations
    2. Averages qt/et times within each history
    3. Calibrates times based on first configuration
    4. Normalizes similarity weights across histories
    
    Args:
        histories_with_weights: Sequence of (History, similarity_weight) tuples
        sql_type: SQL type ('qt' or 'et')
        top_ratio: Ratio of top observations to keep per history (0.0-1.0)
    
    Returns:
        DataFrame with one row per history (aggregated), with normalized weights
    """
    if not histories_with_weights:
        return pd.DataFrame()
    
    valid_histories_weights = [(h, w) for h, w in histories_with_weights if w > 0 and len(h) > 0]
    
    if not valid_histories_weights:
        return pd.DataFrame()
    
    weights = np.array([w for _, w in valid_histories_weights])
    total_weight = weights.sum()
    if total_weight > 0:
        weights = weights / total_weight
    
    reference_history: History = valid_histories_weights[0][0]
    records: List[Dict[str, float]] = []
    for (history, _), normalized_weight in zip(valid_histories_weights, weights):
        # Compute calibration factor, skip first history as reference
        calibration_factor = 1.0
        if reference_history is not None and history is not reference_history:
            calibration_factor = _compute_calibration_factor(history, reference_history, sql_type=sql_type)
        
        aggregated_record = _aggregate_history_records(
            history,
            sql_type=sql_type,
            top_ratio=top_ratio,
            calibration_factor=calibration_factor,
        )
        
        if not aggregated_record:
            continue
        
        aggregated_record["sample_weight"] = float(normalized_weight)
        records.append(aggregated_record)
    
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    sql_prefix = f"{sql_type}_"
    sql_columns = [col for col in df.columns if col.startswith(sql_prefix)]
    
    for col in sql_columns:
        df[col] = df[col].fillna(float("inf"))
    
    df["objective"] = df["objective"].fillna(float("inf"))
    df["elapsed_time"] = df["elapsed_time"].fillna(df["objective"])
    df["sample_weight"] = df["sample_weight"].fillna(0.0)
    
    return df

# ---------------------------------------------------------------------------
# Compute weighted SQL statistics
# ---------------------------------------------------------------------------
def compute_weighted_sql_statistics(
    df: pd.DataFrame,
    sql_columns: Sequence[str],
    *,
    weights: np.ndarray,
    correlation_method: str = "spearman",
    sql_type: str = "qt",
    tolerance: float = 0.1,
) -> Dict[str, Dict[str, float]]:
    """Compute weighted statistics for each SQL candidate.
    
    - df: (n_histories, n_columns)
    - sql_columns: List of SQL column names, (<num_sqls>,)
    - weights: (n_histories,)
    """

    sql_prefix = f"{sql_type}_"

    total_times = df["objective"].to_numpy(copy=True)
    total_times[~np.isfinite(total_times)] = 0.0    # set infinite values to 0.0
    total_weighted_time = float(np.sum(weights * total_times))
    logger.info("compute_weighted_sql_statistics: total_times = %s", total_times)
    logger.info("compute_weighted_sql_statistics: total_weighted_time = %s", total_weighted_time)

    sql_stats: Dict[str, Dict[str, float]] = {}

    for sql_name in sql_columns:
        col = f"{sql_prefix}{sql_name}"
        if col not in df.columns:
            continue

        # (n_histories,) aggregated sql-time of each history for the given SQL
        sql_times = df[col].to_numpy(copy=True)
        # (n_histories,) mask of valid sql-times
        finite_mask = np.isfinite(sql_times) & (weights > 0)
        if not finite_mask.any():
            sql_stats[sql_name] = {
                "estimated_time": 0.0,
                "correlation": 0.0,
                "avg_time": 0.0,
                "total_time": 0.0,
                "tolerance": tolerance,
            }
            continue

        """
        H1	8.0	    0.5
        H2	NaN	    0.3
        H3	14.0	0.2

        weighted_time = 0.5 * 8.0 + 0.2 * 14.0 = 7.2
        avg_weight = 0.5 + 0.2 = 0.7
        avg_time = weighted_time / avg_weight = 7.2 / 0.7 ≈ 10.2857
        """
        weighted_time = float(np.sum(weights[finite_mask] * sql_times[finite_mask]))
        normalized_time = (
            weighted_time / total_weighted_time if total_weighted_time > 0 else 0.0
        )
        avg_weight = weights[finite_mask].sum()
        avg_time = weighted_time / avg_weight if avg_weight > 0 else 0.0

        correlation = _safe_weighted_corr(sql_times, total_times, weights, correlation_method)
        logger.info("compute_weighted_sql_statistics: correlation(%s, %s) = %s", sql_name, "objective", correlation)

        sql_stats[sql_name] = {
            "estimated_time": normalized_time,
            "correlation": correlation,
            "avg_time": avg_time,
            "total_time": weighted_time,
            "tolerance": tolerance,
        }

    return sql_stats


def compute_subset_correlation(
    df: pd.DataFrame,
    selected_queries: Iterable[str],
    *,
    weights: np.ndarray,
    correlation_method: str = "spearman",
    sql_type: str = "qt",
) -> float:
    selected_queries = list(selected_queries)
    if not selected_queries:
        return 0.0

    sql_prefix = f"{sql_type}_"
    subset_times: List[float] = []
    total_times: List[float] = []
    subset_weights: List[float] = []

    for idx, row in df.iterrows():
        weight = weights[idx]
        if weight <= 0:
            continue

        subset_time = 0.0
        valid = False
        for sql_name in selected_queries:
            col = f"{sql_prefix}{sql_name}"
            if col not in row:
                continue
            value = row[col]
            if np.isfinite(value):
                subset_time += float(value)
                valid = True

        if not valid:
            continue

        total_time = row.get("objective", float("inf"))
        if not np.isfinite(total_time) or total_time <= 0:
            continue

        subset_times.append(subset_time)
        total_times.append(float(total_time))
        subset_weights.append(weight)

    if len(subset_times) < 3:
        return 0.0

    return _weighted_corr(
        np.asarray(subset_times, dtype=float),
        np.asarray(total_times, dtype=float),
        np.asarray(subset_weights, dtype=float),
        correlation_method,
    )


def compute_query_similarity(
    df: pd.DataFrame,
    candidate_sql: str,
    selected_queries: Iterable[str],
    *,
    weights: np.ndarray,
    correlation_method: str = "spearman",
    sql_type: str = "qt",
) -> float:
    selected_queries = list(selected_queries)
    if not selected_queries:
        return 0.0

    sql_prefix = f"{sql_type}_"
    target_col = f"{sql_prefix}{candidate_sql}"
    if target_col not in df.columns:
        return 0.0

    candidate_times = df[target_col].to_numpy(copy=True)
    similarities: List[float] = []

    # compute similarity between the candidate SQL and each selected SQL
    for sql_name in selected_queries:
        col = f"{sql_prefix}{sql_name}"
        if col not in df.columns:
            continue
        other_times = df[col].to_numpy(copy=True)
        # compute correlation between the candidate SQL and the selected SQL
        corr = _safe_weighted_corr(candidate_times, other_times, weights, correlation_method)
        # Put the absolute value of the correlation coefficient into the similarities list. 
        # The absolute value is taken because whether the correlation is positive or negative, as long as the absolute value is large, 
        # it means that the change trend of the two SQLs is highly synchronized, which may cause redundancy.
        similarities.append(abs(corr))

    # Return max(similarities) - the correlation between the candidate SQL and the SQL 
    # in the selected queries that is most similar.
    # If the list is empty (no valid comparison), return 0.
    return max(similarities) if similarities else 0.0


def select_sql_subset_for_fidelity(
    df: pd.DataFrame,
    sql_stats: Dict[str, Dict[str, float]],
    fidelity_ratio: float,
    used_queries: Iterable[str],
    *,
    weights: np.ndarray,
    lambda_penalty: float = 0.1,
    correlation_method: str = "spearman",
    sql_type: str = "qt",
    tolerance: float = 0.1,
) -> Tuple[List[str], float]:
    """Greedy weighted selection for a given fidelity ratio."""

    total_estimated_time = sum(stats["estimated_time"] for stats in sql_stats.values())
    budget = fidelity_ratio * total_estimated_time
    max_budget = budget * (1 + tolerance)
    logger.info("select_sql_subset_for_fidelity: total_estimated_time = %s, budget = %s, max_budget = %s", total_estimated_time, budget, max_budget)

    selected: List[str] = []
    current_time = 0.0

    candidate_queries = [q for q in sql_stats.keys() if q not in used_queries]

    iteration = 0
    while current_time < max_budget and candidate_queries:
        iteration += 1
        best_query = None
        best_score = -float("inf")

        for sql_name in candidate_queries:
            if sql_name in selected:
                continue

            marginal_gain = sql_stats[sql_name]["correlation"]
            # compute redundancy between the candidate SQL and the selected SQLs
            redundancy = compute_query_similarity(
                df,
                sql_name,
                selected,
                weights=weights,
                correlation_method=correlation_method,
                sql_type=sql_type,
            )
            # score = marginal_gain - lambda_penalty * redundancy
            score = marginal_gain - lambda_penalty * redundancy

            estimated_time = sql_stats[sql_name]["estimated_time"]
            if current_time + estimated_time <= budget and score > best_score:
                best_query = sql_name
                best_score = score

        if best_query is None:
            break

        selected.append(best_query)
        current_time += sql_stats[best_query]["estimated_time"]
        candidate_queries.remove(best_query)

    total_ratio = current_time / total_estimated_time if total_estimated_time > 0 else 0.0
    logger.info("select_sql_subset_for_fidelity: selected = %s, total_ratio = %s", selected, total_ratio)
    return selected, total_ratio


def multi_fidelity_sql_selection(
    df: pd.DataFrame,
    fidelity_levels: Sequence[float],
    *,
    weights: np.ndarray,
    lambda_penalty: float = 0.1,
    correlation_method: str = "spearman",
    sql_type: str = "qt",
    tolerance: float = 0.1,
) -> Tuple[Dict[float, List[str]], Dict[str, Dict[str, float]]]:
    """Weighted multi-fidelity SQL subset selection.

    Args:
        - df: DataFrame containing the aggregated data (each row is aggregated from multiple observations within one History)
        - fidelity_levels: List of fidelity levels, from 0.0 to 1.0
        - weights: Array of weights, weights[i] is the weight of the i-th aggregated data, the sum of weights is 1.0
        - lambda_penalty: Lambda penalty, a hyperparameter to balance the correlation and redundancy
        - correlation_method: Correlation method, 'spearman' or 'kendall'
        - sql_type: SQL type, 'qt' or 'et'  (qt: query time, et: elapsed time)
        - tolerance: Tolerance, a hyperparameter to balance the time budget

    Returns:
        - fidelity_subsets: Dictionary of fidelity subsets, fidelity_subsets[fidelity] is the list of SQLs for the given fidelity
        - sql_stats: Dictionary of SQL statistics, sql_stats[sql_name] is the statistics of the given SQL
    
    """

    sql_prefix = f"{sql_type}_"
    sql_columns = sorted({col[len(sql_prefix):] for col in df.columns if col.startswith(sql_prefix)})

    if not sql_columns:
        return {}, {}

    sql_stats = compute_weighted_sql_statistics(
        df,
        sql_columns,
        weights=weights,
        correlation_method=correlation_method,
        sql_type=sql_type,
        tolerance=tolerance,
    )

    fidelity_subsets: Dict[float, List[str]] = {}
    used_queries: List[str] = []

    for fidelity in sorted(fidelity_levels):
        if math.isclose(fidelity, 1.0):
            fidelity_subsets[fidelity] = sql_columns.copy()
            continue

        selected, _ = select_sql_subset_for_fidelity(
            df,
            sql_stats,
            fidelity,
            used_queries,
            weights=weights,
            lambda_penalty=lambda_penalty,
            correlation_method=correlation_method,
            sql_type=sql_type,
            tolerance=tolerance,
        )

        fidelity_subsets[fidelity] = selected
        used_queries.extend(selected)

    return fidelity_subsets, sql_stats
