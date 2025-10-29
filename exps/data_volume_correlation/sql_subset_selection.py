"""
Multi-fidelity SQL subset selection algorithm based on historical data

Implements the Weighted Multi-History Multi-Fidelity SQL Subset Selection algorithm
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau
from typing import List, Dict, Set, Tuple, Optional
import json


def _compute_correlation(x, y, method='spearman'):
    if method == 'spearman':
        corr, _ = spearmanr(x, y)
    elif method == 'pearson':
        corr, _ = pearsonr(x, y)
    elif method == 'kendall':
        corr, _ = kendalltau(x, y)
    return abs(corr) if not np.isnan(corr) else 0.0


def _identify_sql_columns(df: pd.DataFrame, sql_type: str = 'qt') -> List[str]:
    sql_prefix = f"{sql_type}_"
    return [col for col in df.columns if col.startswith(sql_prefix)]


def _display_sql_statistics(sql_stats: Dict, top_n: int = 10):
    sorted_sqls = sorted(sql_stats.items(), key=lambda x: x[1]['correlation'], reverse=True)
    print(f"\nTop {top_n} highest correlation SQLs:")
    for i, (sql, stats) in enumerate(sorted_sqls[:top_n], 1):
        print(f"  {i}. {sql}: correlation={stats['correlation']:.4f}, time_ratio={stats['estimated_time']:.4f}")


def _validate_parameters(fidelity_levels: List[float], correlation_method: str, 
                        time_type: str, sql_type: str, tolerance: float):
    if not fidelity_levels:
        raise ValueError("fidelity_levels cannot be empty")
    
    if not all(0 < f <= 1.0 for f in fidelity_levels):
        raise ValueError("All fidelity levels must be between 0 and 1")
    
    if correlation_method not in ['spearman', 'pearson', 'kendall']:
        raise ValueError(f"Unknown correlation method: {correlation_method}")
    
    if time_type not in ['spark_time', 'elapsed_time']:
        raise ValueError(f"Unknown time type: {time_type}")
    
    if sql_type not in ['qt', 'et']:
        raise ValueError(f"Unknown SQL type: {sql_type}")
    
    if not 0 <= tolerance <= 1:
        raise ValueError("Tolerance must be between 0 and 1")


def compute_sql_correlation(df, sql_col, target_col='objective', method='spearman'):
    # Filter valid data
    valid_mask = (df[sql_col] != float('inf')) & (df[target_col] != float('inf'))
    valid_df = df[valid_mask]
    
    if len(valid_df) < 3:
        return 0.0
    
    return _compute_correlation(valid_df[sql_col], valid_df[target_col], method)


def compute_weighted_sql_statistics(df, sql_columns, similarity_weight=1.0, correlation_method='spearman', 
                                   time_type='spark_time', sql_type='qt', tolerance=0.1):
    """
    Compute weighted SQL statistics
    
    Args:
        df: DataFrame containing fidelity=1.0 data
        sql_columns: List of SQL column names
        similarity_weight: Similarity weight (currently set to 1)
        correlation_method: Correlation calculation method
        time_type: Time type ('spark_time' or 'elapsed_time')
        sql_type: SQL type ('qt' or 'et')
        tolerance: Time budget tolerance
        
    Returns:
        Dictionary containing weighted execution time and correlation
    """
    total_time = df[time_type].sum()
    
    sql_stats = {}
    
    for sql_col in sql_columns:
        sql_data = df[sql_col].replace(float('inf'), pd.NA)
        sql_avg_time = sql_data.mean() if not sql_data.isna().all() else 0
        sql_total_time = df[sql_col].replace(float('inf'), 0).sum()
        normalized_time = sql_total_time / total_time if total_time > 0 else 0
        
        correlation = compute_sql_correlation(df, sql_col, 'objective', correlation_method)
        
        sql_stats[sql_col] = {
            'estimated_time': normalized_time,
            'correlation': correlation,
            'avg_time': sql_avg_time,
            'total_time': sql_total_time,
            'tolerance': tolerance
        }
    
    return sql_stats


def compute_subset_correlation(df, selected_queries, sql_stats, correlation_method='spearman'):
    """
    Calculate correlation between selected SQL subset total time and all SQL total time
    
    Args:
        df: DataFrame containing all SQL performance data
        selected_queries: Set of selected SQL queries
        sql_stats: SQL statistics
        correlation_method: Correlation calculation method
        
    Returns:
        Correlation coefficient
    """
    if not selected_queries:
        return 0.0
    
    subset_times = []
    total_times = []
    
    for _, row in df.iterrows():
        subset_time = 0.0
        for sql in selected_queries:
            if sql in row and row[sql] != float('inf'):
                subset_time += row[sql]
        
        total_time = row.get('spark_time', 0.0)
        if total_time == 0 or total_time == float('inf'):
            continue
            
        subset_times.append(subset_time)
        total_times.append(total_time)
    
    return _compute_correlation(subset_times, total_times, correlation_method)


def compute_query_similarity(q, selected_queries, df, sql_columns):
    """
    Calculate similarity between query and selected query set (based on execution time patterns)
    
    Args:
        q: Query column name
        selected_queries: Set of selected queries
        df: DataFrame
        sql_columns: All SQL column names
        
    Returns:
        Similarity score (0-1)
    """
    if len(selected_queries) == 0:
        return 0.0
    
    q_times = df[q].replace(float('inf'), 0).values
    
    similarities = []
    for sq in selected_queries:
        sq_times = df[sq].replace(float('inf'), 0).values
        
        valid_mask = (q_times != 0) & (sq_times != 0)
        if valid_mask.sum() < 3:
            similarities.append(0.0)
            continue
        
        try:
            corr, _ = spearmanr(q_times[valid_mask], sq_times[valid_mask])
            similarities.append(abs(corr) if not np.isnan(corr) else 0.0)
        except:
            similarities.append(0.0)
    return max(similarities) if similarities else 0.0


def compute_marginal_gain(q, selected_set, sql_stats, df):
    """
    Calculate marginal gain after adding query q
    
    Args:
        q: Candidate query
        selected_set: Selected query set
        sql_stats: SQL statistics
        df: DataFrame
        
    Returns:
        Marginal gain
    """
    current_correlation = sum(sql_stats[sq]['correlation'] for sq in selected_set) if selected_set else 0.0
    new_correlation = current_correlation + sql_stats[q]['correlation']
    marginal_gain = new_correlation - current_correlation
    return marginal_gain


def select_sql_subset_for_fidelity(
    sql_stats: Dict,
    fidelity_ratio: float,
    used_queries: Set[str],
    df: pd.DataFrame,
    sql_columns: List[str],
    lambda_penalty: float = 0.1,
    correlation_method: str = 'spearman',
    tolerance: float = 0.1
) -> Tuple[Set[str], float]:
    """
    Select SQL subset for specific fidelity level
    
    Args:
        sql_stats: SQL statistics dictionary
        fidelity_ratio: Fidelity ratio (e.g., 0.1, 0.3, etc.)
        used_queries: Set of already used queries
        df: DataFrame
        sql_columns: All SQL column names
        lambda_penalty: Redundancy penalty parameter
        
    Returns:
        (Selected SQL set, actual time budget ratio used)
    """
    total_estimated_time = sum(stats['estimated_time'] for stats in sql_stats.values())
    budget = fidelity_ratio * total_estimated_time
    max_budget = budget * (1 + tolerance)
    
    selected_queries = set()
    current_time = 0.0
    
    candidate_queries = [q for q in sql_columns if q not in used_queries]
    
    print(f"\nFidelity level {fidelity_ratio}:")
    print(f"  Time budget: {budget:.4f} (max: {max_budget:.4f})")
    print(f"  Candidate queries: {len(candidate_queries)}")
    
    iteration = 0
    while current_time < max_budget and candidate_queries:
        iteration += 1
        best_query = None
        best_score = -float('inf')
        
        for q in candidate_queries:
            if q in selected_queries:
                continue
            
            # Calculate marginal gain
            marginal_gain = compute_marginal_gain(q, selected_queries, sql_stats, df)
            
            # Calculate redundancy
            redundancy = compute_query_similarity(q, selected_queries, df, sql_columns)
            
            # Calculate comprehensive score
            score = marginal_gain - lambda_penalty * redundancy
            
            # Check if within budget
            if current_time + sql_stats[q]['estimated_time'] <= budget:
                if score > best_score:
                    best_score = score
                    best_query = q
        
        if best_query is None:
            break
        
        selected_queries.add(best_query)
        current_time += sql_stats[best_query]['estimated_time']
        candidate_queries.remove(best_query)
        
        subset_correlation = compute_subset_correlation(df, selected_queries, sql_stats, correlation_method)
        
        print(f"  Iteration {iteration}: Selected {best_query}, score={best_score:.4f}, "
              f"time={sql_stats[best_query]['estimated_time']:.4f}, "
              f"cumulative={current_time:.4f}")
        print(f"    Current set correlation with total SQL: {subset_correlation:.4f}")
    
    print(f"  Final selection: {len(selected_queries)} queries, time ratio: {current_time:.4f}/{budget:.4f}")
    
    return selected_queries, current_time / total_estimated_time if total_estimated_time > 0 else 0.0


def multi_fidelity_sql_selection(
    df: pd.DataFrame,
    fidelity_levels: List[float] = [0.1, 0.3, 0.6, 1.0],
    lambda_penalty: float = 0.1,
    correlation_method: str = 'spearman',
    time_type: str = 'spark_time',
    sql_type: str = 'qt',
    tolerance: float = 0.05
) -> Tuple[Dict[float, Set[str]], Dict]:
    """
    Multi-fidelity SQL subset selection algorithm
    
    Args:
        df: DataFrame containing fidelity=1.0 data
        fidelity_levels: List of fidelity levels
        lambda_penalty: Redundancy penalty parameter
        time_type: Time type ('spark_time' or 'elapsed_time')
        sql_type: SQL type ('qt' or 'et')
        tolerance: Time budget tolerance
        
    Returns:
        Tuple of (fidelity_subsets, sql_stats)
    """
    _validate_parameters(fidelity_levels, correlation_method, time_type, sql_type, tolerance)
    
    sql_columns = _identify_sql_columns(df, sql_type)
    
    print(f"Found {len(sql_columns)} SQL query columns (type: {sql_type})")
    print(f"SQL columns: {sql_columns[:10]}...")
    
    print(f"\nComputing SQL statistics (using {correlation_method} correlation)...")
    sql_stats = compute_weighted_sql_statistics(df, sql_columns, correlation_method=correlation_method, 
                                            time_type=time_type, sql_type=sql_type, tolerance=tolerance)
    
    _display_sql_statistics(sql_stats)
    
    fidelity_subsets = {}
    used_queries = set()
    
    print("\n" + "=" * 60)
    print("Starting multi-fidelity SQL subset selection (non-incremental)")
    print("=" * 60)
    
    for fidelity in sorted(fidelity_levels):
        if fidelity == 1.0:
            fidelity_subsets[fidelity] = set(sql_columns)
            print(f"\nFidelity level 1.0: Using all {len(sql_columns)} queries")
        else:
            selected, actual_ratio = select_sql_subset_for_fidelity(
                sql_stats, fidelity, used_queries, df, sql_columns, lambda_penalty, correlation_method, tolerance
            )
            fidelity_subsets[fidelity] = selected
            used_queries.update(selected)
    
    return fidelity_subsets, sql_stats


def multi_fidelity_sql_selection_incremental(
    df: pd.DataFrame,
    fidelity_levels: List[float] = [0.1, 0.3, 0.6, 1.0],
    lambda_penalty: float = 0.1,
    correlation_method: str = 'spearman',
    time_type: str = 'spark_time',
    sql_type: str = 'qt',
    tolerance: float = 0.1
) -> Tuple[Dict[float, Set[str]], Dict]:
    """
    Multi-fidelity SQL subset selection algorithm (incremental version)
    Lower fidelity is a subset of higher fidelity, can reuse low fidelity results
    
    Args:
        df: DataFrame containing fidelity=1.0 data (each row is a configuration's performance)
        fidelity_levels: List of fidelity levels
        lambda_penalty: Redundancy penalty parameter
        correlation_method: Correlation calculation method
        time_type: Time type ('spark_time' or 'elapsed_time')
        sql_type: SQL type ('qt' or 'et')
        tolerance: Time budget tolerance
        
    Returns:
        fidelity_subsets: SQL subset for each fidelity level
        sql_stats: SQL statistics
    """
    _validate_parameters(fidelity_levels, correlation_method, time_type, sql_type, tolerance)
    
    sql_columns = _identify_sql_columns(df, sql_type)
    
    print(f"Found {len(sql_columns)} SQL query columns")
    print(f"SQL columns: {sql_columns[:10]}...")
    
    print(f"\nComputing SQL statistics (using {correlation_method} correlation)...")
    sql_stats = compute_weighted_sql_statistics(df, sql_columns, correlation_method=correlation_method,
                                              time_type=time_type, sql_type=sql_type, tolerance=tolerance)

    _display_sql_statistics(sql_stats)

    fidelity_subsets = {}
    current_sqls = set()
    
    print("\n" + "=" * 60)
    print("Starting multi-fidelity SQL subset selection (incremental version)")
    print("=" * 60)
    
    for fidelity in sorted(fidelity_levels):
        if fidelity == 1.0:
            fidelity_subsets[fidelity] = set(sql_columns)
            print(f"\nFidelity level 1.0: Using all {len(sql_columns)} queries")
        else:
            target_ratio = fidelity
            current_ratio = sum(sql_stats[sql]['estimated_time'] for sql in current_sqls)
            remaining_budget = target_ratio - current_ratio
            
            if remaining_budget <= 0:
                fidelity_subsets[fidelity] = current_sqls.copy()
                print(f"\nFidelity level {fidelity}: Using current {len(current_sqls)} queries (target ratio reached)")
            else:
                remaining_sqls = [sql for sql in sql_columns if sql not in current_sqls]
                
                if not remaining_sqls:
                    fidelity_subsets[fidelity] = current_sqls.copy()
                    print(f"\nFidelity level {fidelity}: Using current {len(current_sqls)} queries (no remaining SQLs)")
                else:
                    selected_new, actual_ratio = select_sql_subset_for_fidelity(
                        sql_stats, remaining_budget, set(), df, remaining_sqls, lambda_penalty, correlation_method
                    )
                    
                    current_sqls.update(selected_new)
                    fidelity_subsets[fidelity] = current_sqls.copy()
                    
                    print(f"\nFidelity level {fidelity}: Added {len(selected_new)} queries, total {len(current_sqls)} queries")
                    print(f"  New SQLs: {sorted(list(selected_new))}")
    
    return fidelity_subsets, sql_stats


def analyze_fidelity_subsets(fidelity_subsets: Dict[float, Set[str]], sql_stats: Dict, df: pd.DataFrame, correlation_method: str = 'spearman'):
    """
    Analyze fidelity subset statistics
    
    Args:
        fidelity_subsets: Fidelity subset dictionary
        sql_stats: SQL statistics
        df: DataFrame containing all SQL performance data
        correlation_method: Correlation calculation method
    """
    print("\n" + "=" * 60)
    print("Fidelity subset analysis")
    print("=" * 60)
    
    for fidelity in sorted(fidelity_subsets.keys()):
        subset = fidelity_subsets[fidelity]
        
        total_time = sum(sql_stats[q]['estimated_time'] for q in subset)
        subset_correlation = compute_subset_correlation(df, subset, sql_stats, correlation_method)
        
        correlations = [sql_stats[q]['correlation'] for q in subset]
        avg_individual_corr = sum(correlations) / len(correlations) if correlations else 0.0
        max_corr = max(correlations) if correlations else 0.0
        min_corr = min(correlations) if correlations else 0.0
        std_corr = np.std(correlations) if len(correlations) > 1 else 0.0
        
        print(f"\nFidelity level {fidelity}:")
        print(f"  Query count: {len(subset)}")
        print(f"  Total time ratio: {total_time:.4f}")
        print(f"  Set correlation with total SQL: {subset_correlation:.4f}")
        print(f"  Individual SQL correlation statistics:")
        print(f"    Average correlation: {avg_individual_corr:.4f}")
        print(f"    Highest correlation: {max_corr:.4f}")
        print(f"    Lowest correlation: {min_corr:.4f}")
        print(f"    Standard deviation: {std_corr:.4f}")
        
        if len(subset) <= 20:
            print(f"  Detailed correlations:")
            sorted_sqls = sorted([(q, sql_stats[q]['correlation']) for q in subset], 
                               key=lambda x: x[1], reverse=True)
            for sql, corr in sorted_sqls:
                print(f"    {sql}: {corr:.4f}")
        else:
            sorted_sqls = sorted([(q, sql_stats[q]['correlation']) for q in subset], 
                               key=lambda x: x[1], reverse=True)
            print(f"  Top 10 correlations:")
            for sql, corr in sorted_sqls[:10]:
                print(f"    {sql}: {corr:.4f}")
            if len(sorted_sqls) > 10:
                print(f"  ... (omitted {len(sorted_sqls)-10} items)")
                print(f"  Bottom 5 correlations:")
                for sql, corr in sorted_sqls[-5:]:
                    print(f"    {sql}: {corr:.4f}")

