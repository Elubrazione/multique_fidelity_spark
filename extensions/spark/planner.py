import json
from typing import Dict, Iterable, List, Optional, Tuple
from openbox import logger

from Evaluator import Planner
from .partitioner import SQLPartitioner, PartitionPlan


class SparkSQLPlanner(Planner[PartitionPlan]):
    def __init__(
        self,
        partitioner: SQLPartitioner,
        *,
        timeout: Optional[Dict[str, float]] = None,
        fallback_sqls: Optional[Dict[float, Iterable[str]]] = None,
    ) -> None:
        super().__init__()
        self.partitioner = partitioner
        self.timeout = timeout or {}
        self.fallback_sqls = {
            fidelity: list(sqls)
            for fidelity, sqls in (fallback_sqls or {}).items()
        }

    def refresh_plan(self, *, force: bool = False) -> PartitionPlan:
        plan = self.partitioner.refresh_from_task_manager(force=force)
        self._cached_plan = plan
        fidelity_snapshot = {
            f"{float(fid):.5f}": list(sqls)
            for fid, sqls in plan.fidelity_subsets.items()
        }
        logger.info(
            "SparkSQLPlanner: refreshed plan (force=%s) with fidelity subsets: %s",
            force,
            json.dumps(fidelity_snapshot, ensure_ascii=False),
        )
        return plan

    def plan(
        self,
        resource_ratio: float,
        **kwargs
    ) -> Optional[Dict[str, Iterable[str]]]:
        resource_ratio = round(float(resource_ratio), 5)
        plan = self._ensure_plan(
            force_refresh=kwargs.get('force_refresh', False),
            check_dirty=lambda: self.partitioner.is_plan_dirty(),
        )
        
        if plan is None:
            logger.warning("SparkSQLPlanner: failed to get plan")
            return None

        subset, fidelity_used = self._lookup_sql_subset(plan, resource_ratio)
        plan_source = "partition"

        if subset is None and kwargs.get('allow_fallback', True):
            fallback_subset, fallback_fidelity = self._lookup_fallback(resource_ratio)
            if fallback_subset is not None:
                subset = fallback_subset
                fidelity_used = fallback_fidelity
                plan_source = "fallback"

        if subset is None:
            return None

        timeouts = {sql: self.timeout.get(sql) for sql in subset if sql in self.timeout}

        return {
            "sqls": subset,
            "timeout": timeouts,
            "selected_fidelity": float(fidelity_used),
            "plan_source": plan_source,
        }

    def _lookup_sql_subset(
        self,
        plan: PartitionPlan,
        resource_ratio: float,
    ) -> Tuple[Optional[List[str]], Optional[float]]:
        if not plan.fidelity_subsets:
            logger.warning(f"No plan found for resource ratio {resource_ratio}")
            return None, None

        if resource_ratio in plan.fidelity_subsets:
            logger.debug(f"Found plan for resource ratio {resource_ratio}")
            return list(plan.fidelity_subsets[resource_ratio]), resource_ratio

        return None, None

    def _lookup_fallback(self, resource_ratio: float) -> Tuple[Optional[List[str]], Optional[float]]:
        logger.warning(f"Lookup fallback sqls for resource ratio {resource_ratio}")
        if not self.fallback_sqls:
            return None, None

        if resource_ratio in self.fallback_sqls:
            return list(self.fallback_sqls[resource_ratio]), resource_ratio

        return None, None

