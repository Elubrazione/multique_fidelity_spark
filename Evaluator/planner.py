import json
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Tuple
from openbox import logger

from .partitioner import PartitionPlan, SQLPartitioner


class Planner(ABC):
    @abstractmethod
    def plan(self, resource_ratio: float, **kwargs) -> dict:
        pass


class SparkSQLPlanner(Planner):
    def __init__(
        self,
        partitioner: SQLPartitioner,
        *,
        timeout: Optional[Dict[str, float]] = None,
        fallback_sqls: Optional[Dict[float, Iterable[str]]] = None,
    ) -> None:
        self.partitioner = partitioner
        self.timeout = timeout or {}
        self.fallback_sqls = {
            fidelity: list(sqls)
            for fidelity, sqls in (fallback_sqls or {}).items()
        }
        self._cached_plan: Optional[PartitionPlan] = None

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
        *,
        force_refresh: bool = False,
        allow_fallback: bool = True,
    ) -> Dict[str, Iterable[str]]:
        resource_ratio = round(float(resource_ratio), 5)
        plan = self._ensure_plan(force_refresh=force_refresh)

        subset, fidelity_used = self._lookup_sql_subset(plan, resource_ratio)
        plan_source = "partition"

        if subset is None and allow_fallback:
            fallback_subset, fallback_fidelity = self._lookup_fallback(resource_ratio)
            if fallback_subset is not None:
                subset = fallback_subset
                fidelity_used = fallback_fidelity
                plan_source = "fallback"

        timeouts = {sql: self.timeout.get(sql) for sql in subset if sql in self.timeout}

        return {
            "sqls": subset,
            "timeout": timeouts,
            "selected_fidelity": float(fidelity_used),
            "plan_source": plan_source,
        }

    def _ensure_plan(self, *, force_refresh: bool) -> PartitionPlan:
        if force_refresh:
            logger.warning("Force refreshing planner plan")
            return self.refresh_plan(force=True)

        if self._cached_plan is None:
            logger.warning("No cached plan, refreshing from task manager")
            return self.refresh_plan(force=True)

        if self.partitioner.is_plan_dirty():
            logger.warning("Planner plan is dirty, refreshing from task manager")
            return self.refresh_plan(force=True)

        return self._cached_plan

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