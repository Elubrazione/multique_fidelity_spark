from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
from openbox.utils.history import History, Observation


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
            - spark_time: float, sum of all query times
            - elapsed_time: float
            - {sql_type}_{query_name}: float, for each query in sql_times
    """

    extra_info = getattr(observation, "extra_info", None) or {}
    sql_times: Dict[str, float] = extra_info.get(f"{sql_type}_time", {})

    record: Dict[str, float] = {}

    objectives = getattr(observation, "objectives", None)
    if objectives:
        record["objective"] = float(objectives[0])

    spark_like_time = 0.0
    for sql_name, value in sql_times.items():
        column_name = f"{sql_type}_{sql_name}"
        record[column_name] = float(value)
        if np.isfinite(value):
            spark_like_time += float(value)

    record["spark_time"] = spark_like_time if spark_like_time > 0 else float("inf")

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
    
    ref_spark_time = ref_record.get("spark_time", float("inf"))
    spark_time = record.get("spark_time", float("inf"))
    
    if not np.isfinite(ref_spark_time) or not np.isfinite(spark_time) or spark_time <= 0:
        return 1.0
    
    return float(ref_spark_time / spark_time)


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
        
        if "spark_time" in record and np.isfinite(record["spark_time"]):
            spark_times.append(float(record["spark_time"]) * calibration_factor)
        if "objective" in record and np.isfinite(record["objective"]):
            objectives.append(float(record["objective"]) * calibration_factor)
        if "elapsed_time" in record and np.isfinite(record["elapsed_time"]):
            elapsed_times.append(float(record["elapsed_time"]) * calibration_factor)
    
    aggregated: Dict[str, float] = {}
    for sql_name, times in sql_times_dict.items():
        aggregated[f"{sql_prefix}{sql_name}"] = float(np.mean(times)) if times else float("inf")
    aggregated["spark_time"] = float(np.mean(spark_times)) if spark_times else float("inf")
    aggregated["objective"] = float(np.mean(objectives)) if objectives else float("inf")
    aggregated["elapsed_time"] = float(np.mean(elapsed_times)) if elapsed_times else aggregated["spark_time"]

    return aggregated


def build_weighted_dataframe(
    histories_with_weights: Sequence[Tuple[History, float]],
    sql_type: str = "qt",
    top_ratio: float = 1.0,
    normalize_similarities: bool = True,
    enable_calibration: bool = True,
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
        normalize_similarities: Whether to normalize similarity weights to sum to 1.0
        enable_calibration: Whether to enable time calibration based on first config
    
    Returns:
        DataFrame with one row per history (aggregated), with normalized weights
    """
    if not histories_with_weights:
        return pd.DataFrame()
    
    valid_histories_weights = [(h, w) for h, w in histories_with_weights if w > 0 and len(h) > 0]
    
    if not valid_histories_weights:
        return pd.DataFrame()
    
    weights = np.array([w for _, w in valid_histories_weights])
    if normalize_similarities:
        total_weight = weights.sum()
        if total_weight > 0:
            weights = weights / total_weight
    
    reference_history: History = valid_histories_weights[0][0]
    records: List[Dict[str, float]] = []
    for (history, _), normalized_weight in zip(valid_histories_weights, weights):
        # Compute calibration factor, skip first history as reference
        calibration_factor = 1.0
        if enable_calibration and reference_history is not None and history is not reference_history:
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
    
    if "objective" not in df.columns:
        df["objective"] = float("inf")
    else:
        df["objective"] = df["objective"].fillna(float("inf"))
    
    if "spark_time" not in df.columns or df["spark_time"].isna().all():
        df["spark_time"] = df.apply(
            lambda row: float(np.sum([
                value for col in sql_columns
                for value in [row.get(col, float("inf"))]
                if np.isfinite(value)
            ]) or np.inf),
            axis=1,
        )
    else:
        df["spark_time"] = df["spark_time"].fillna(float("inf"))
    
    if "elapsed_time" not in df.columns:
        df["elapsed_time"] = df["spark_time"]
    else:
        df["elapsed_time"] = df["elapsed_time"].fillna(df["spark_time"])
    
    df["sample_weight"] = df["sample_weight"].fillna(0.0)
    
    return df

