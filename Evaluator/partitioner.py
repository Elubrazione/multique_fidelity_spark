from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from openbox import logger
from openbox.utils.history import History

from task_manager import TaskManager
from .utils import (
    build_weighted_dataframe,
    compute_subset_correlation,
    multi_fidelity_sql_selection,
)
from utils.spark import get_full_queries_tasks


@dataclass
class PartitionPlan:
    fidelity_subsets: Dict[float, List[str]] = field(default_factory=dict)
    sql_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)


class SQLPartitioner:
    def __init__(
        self,
        *,
        sql_dir: Optional[str] = None,
        correlation_method: str = "spearman",
        sql_type: str = "qt",   # qt: query time, et: elapsed time
        tolerance: float = 0.1,
        lambda_penalty: float = 0.1,
        current_task_weight: float = 1.0,
        top_ratio: float = 1.0,
        custom_sqls: Optional[List[str]] = None,
    ):
        self.task_manager = TaskManager.instance()
        self.scheduler = self.task_manager.get_scheduler()

        self._default_fidelity_levels = [round(float(1.0), 5)]

        self.correlation_method = correlation_method
        self.sql_type = sql_type
        self.tolerance = tolerance
        self.lambda_penalty = lambda_penalty
        self.current_task_weight = current_task_weight

        self.top_ratio = top_ratio

        self.sql_dir = sql_dir
        self._all_sqls = self._load_all_sqls()
        
        # Custom SQL list (if specified, use this instead of building plan)
        self.custom_sqls: Optional[List[str]] = custom_sqls

        self._latest_plan: Optional[PartitionPlan] = None
        self._plan_dirty: bool = True


    def build_plan(
        self,
        *,
        include_current_task: bool = True,
    ) -> PartitionPlan:
        histories_with_weights = self._collect_histories(include_current_task=include_current_task)

        if not histories_with_weights:
            logger.warning("SQLPartitioner: no histories available to build partition plan.")
            return self._create_empty_plan("no_histories")

        df = build_weighted_dataframe(
            histories_with_weights,
            sql_type=self.sql_type,
            top_ratio=self.top_ratio
        )
        if df.empty:
            logger.warning("SQLPartitioner: aggregated dataframe is empty.")
            return self._create_empty_plan("empty_dataframe")

        weights = df["sample_weight"].to_numpy(dtype=float)
        if not np.any(weights > 0):
            logger.warning("SQLPartitioner: all sample weights are non-positive.")
            return self._create_empty_plan("invalid_weights")

        fidelity_seq = self._resolve_fidelity_levels()

        fidelity_subsets, sql_stats = multi_fidelity_sql_selection(
            df,
            fidelity_seq,
            weights=weights,
            lambda_penalty=self.lambda_penalty,
            correlation_method=self.correlation_method,
            sql_type=self.sql_type,
            tolerance=self.tolerance,
        )

        subset_correlations: Dict[float, float] = {}
        for fidelity, subset in fidelity_subsets.items():
            correlation = compute_subset_correlation(
                df,
                subset,
                weights=weights,
                correlation_method=self.correlation_method,
                sql_type=self.sql_type,
            ) if subset else 0.0
            subset_correlations[fidelity] = correlation

        plan = PartitionPlan(
            fidelity_subsets={k: sorted(values) for k, values in fidelity_subsets.items()},
            sql_stats=sql_stats,
            metadata={
                "histories": [history.task_id for history, _ in histories_with_weights],
                "weights": [weight for _, weight in histories_with_weights],
                "subset_correlation": {
                    float(fidelity): corr for fidelity, corr in subset_correlations.items()
                },
            },
        )

        self._latest_plan = plan
        self._plan_dirty = False
        summary = ", ".join(
            f"{float(fid):.3f} -> {len(sqls)} SQLs" for fid, sqls in plan.fidelity_subsets.items()
        )
        logger.info("SQLPartitioner: built plan with fidelities: %s", summary or "<empty>")
        for fidelity, corr in subset_correlations.items():
            logger.info(
                "SQLPartitioner: fidelity %.3f subset correlation = %.4f",
                float(fidelity),
                corr,
            )
        return plan

    def refresh_from_task_manager(self, *, force: bool = False) -> PartitionPlan:
        # If custom_sqls is specified, create a plan with custom SQLs
        if self.custom_sqls is not None:
            if self._latest_plan is None or force:
                custom_plan = PartitionPlan(
                    fidelity_subsets={1.0: sorted(self.custom_sqls)},
                    sql_stats={},
                    metadata={
                        "custom_sqls": True,
                        "total_queries": len(self.custom_sqls)
                    }
                )
                self._latest_plan = custom_plan
                self._plan_dirty = False
                logger.info(f"SQLPartitioner: Using custom SQL list with {len(self.custom_sqls)} queries")
            return self._latest_plan
        
        # Otherwise, use normal plan building logic
        if force or self._plan_dirty or self._latest_plan is None:
            logger.warning("SQLPartitioner: plan is dirty or latest plan is None, building new plan")
            return self.build_plan(include_current_task=False)
        return self._latest_plan
    
    def set_custom_sqls(self, sqls: Optional[List[str]]):
        """
        Set custom SQL list. If specified, partitioner will use this list
        instead of building plan from history.
        
        Args:
            sqls: List of SQL query names, or None to use normal plan building
        """
        self.custom_sqls = sqls
        self._plan_dirty = True  # Mark as dirty to force refresh
        logger.info(f"SQLPartitioner: Custom SQL list {'set' if sqls else 'cleared'} "
                   f"({len(sqls) if sqls else 0} queries)")

    def mark_plan_dirty(self) -> None:
        self._plan_dirty = True

    def is_plan_dirty(self) -> bool:
        return self._plan_dirty

    @property
    def latest_plan(self) -> Optional[PartitionPlan]:
        return self._latest_plan

    def _collect_histories(
        self,
        *,
        include_current_task: bool,
    ) -> List[Tuple[History, float]]:
        histories: List[Tuple[History, float]] = []

        if include_current_task:
            current_history = self.task_manager.get_current_task_history()
            if current_history is not None and len(current_history) > 0:
                histories.append((current_history, self.current_task_weight))

        similar_histories, similar_scores = self.task_manager.get_similar_tasks()
        if similar_histories:
            for history, (_, score) in zip(similar_histories, similar_scores):
                weight = float(score)
                if len(history) > 0:
                    histories.append((history, weight))

        return histories

    def _create_empty_plan(self, reason: str) -> PartitionPlan:
        fidelity_subsets = {round(float(1.0), 5): self.get_all_sqls()}
        empty_plan = PartitionPlan(fidelity_subsets=fidelity_subsets, metadata={"reason": reason})
        self._latest_plan = empty_plan
        self._plan_dirty = False
        logger.warning("SQLPartitioner: returning fallback plan (%s) with %d SQLs", reason, len(fidelity_subsets.get(1.0, [])))
        return empty_plan

    def _resolve_fidelity_levels(self) -> List[float]:
        scheduler = self.scheduler or self.task_manager.get_scheduler()
        if scheduler is not None:
            try:
                levels = scheduler.get_fidelity_levels()
                if levels:
                    self.scheduler = scheduler
                    return list(levels)
            except Exception as exc:
                logger.warning(f"SQLPartitioner: failed to fetch fidelity levels from scheduler ({exc}); using defaults")
        return list(self._default_fidelity_levels)

    def get_all_sqls(self) -> List[str]:
        return list(self._all_sqls)

    def _load_all_sqls(self) -> List[str]:
        if not self.sql_dir:
            logger.warning("SQLPartitioner: sql_dir not provided; returning empty SQL list")
            return []

        try:
            return get_full_queries_tasks(self.sql_dir)
        except Exception as exc:
            logger.error(f"SQLPartitioner: failed to enumerate SQL directory {self.sql_dir}: {exc}")
            return []

