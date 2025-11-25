import copy
import time
import traceback
import threading
from queue import Queue
from typing import Optional, List
from openbox import logger
from ConfigSpace import Configuration

from .base import Evaluator, NoOpEvaluator
from .planner import Planner, NoOpPlanner
from .partitioner import Partitioner
from .mock_executor import MockExecutor


_DEFAULT_EXTRA_INFO = {}


def _create_default_result(start_time):
    return {
        'result': {'objective': float('inf')},
        'timeout': True,
        'traceback': None,
        'elapsed_time': time.time() - start_time,
        'extra_info': copy.deepcopy(_DEFAULT_EXTRA_INFO),
    }


class EvaluatorManager:
    def __init__(self,
                config_space,
                evaluators: Optional[List[Evaluator]] = None,
                planner: Optional[Planner] = None,
                partitioner: Optional[Partitioner] = None,
                num_evaluator: Optional[int] = 1,
                test_mode: bool = True,
                seed: Optional[int] = None,
                config_manager=None,
                **kwargs
    ):
        """Initialize evaluator manager
        
        Args:
            config_space: Configuration space

            evaluators: Evaluator instances (optional)
            planner: Planner instance (optional)
            partitioner: Partitioner instance (optional)

            num_evaluator: Number of concurrent evaluators
            test_mode: Whether to use test mode (MockExecutor)
            seed: Random seed
            config_manager: Optional config manager for extensions
            **kwargs: Reserved for extensions/custom logic
        """
        self.config_space = config_space
        self.config_manager = config_manager
        self.test_mode = test_mode

        self.evaluators = self._resolve_evaluators(
            evaluators=evaluators,
            test_mode=test_mode,
            seed=seed,
            num_evaluator=num_evaluator
        )
        self.planner = planner or NoOpPlanner()
        self.partitioner = partitioner
        
        # Initialize evaluator pool for concurrent execution
        self.evaluator_queue = Queue()
        for idx in range(len(self.evaluators)):
            self.evaluator_queue.put(idx)

    def _resolve_evaluators(
        self,
        *,
        evaluators: Optional[List[Evaluator]],
        test_mode: bool,
        seed: Optional[int],
        num_evaluator: int,
    ) -> List[Evaluator]:
        if evaluators:
            return evaluators
        if test_mode:
            if seed is not None:
                base_seed = int(seed) + time.time_ns()
            else:
                base_seed = time.time_ns()
            return [MockExecutor(seed=base_seed + i) for i in range(num_evaluator)]
        logger.warning("No evaluator provided; defaulting to NoOpEvaluator.")
        return [NoOpEvaluator() for _ in range(num_evaluator)]
            
    def __call__(self, config: Configuration, resource_ratio: float):
        idx = self.evaluator_queue.get()  # Block until a free evaluator is available
        logger.debug(f"Got free evaluator: {idx}")

        plan = None
        try:
            if self.planner is not None:
                plan = self.planner.plan(resource_ratio, force_refresh=False, allow_fallback=True)
        except Exception as exc:
            logger.error(f"[EvaluatorManager] Failed to obtain evaluation plan for resource {resource_ratio}: {exc}")
            logger.debug(traceback.format_exc())

        if plan is None:
            plan = self._build_fallback_plan(resource_ratio)

        result_queue = Queue()

        def run():
            start_time = time.time()
            result = None
            try:
                result = self.evaluators[idx](config, resource_ratio, plan=plan)
            except Exception as e:
                result = _create_default_result(start_time)
                logger.error(f"[Evaluator {idx}] Execution raised exception, continue with INF objective. Exception: {type(e).__name__}: {str(e)}")
            finally:
                if result is not None:
                    result_queue.put(result)
                else:
                    result = _create_default_result(start_time)
                    result_queue.put(result)
                    logger.error(f"[Evaluator {idx}] Result was None, using default INF result.")
                self.evaluator_queue.put(idx)  # Mark as free again
                logger.debug(f"[Evaluator {idx}] Marked as free again.")

        thread = threading.Thread(target=run)
        thread.start()

        result = result_queue.get()  # Wait for result
        thread.join()
        return result

    def _build_fallback_plan(self, resource_ratio: float):
        # Hook for subclasses/extensions to provide fallback plans.
        return None
