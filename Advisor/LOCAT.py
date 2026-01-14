import numpy as np
import copy
from typing import List, Optional
from openbox import logger
from ConfigSpace import Configuration, ConfigurationSpace

from .base import BaseAdvisor
from .utils import build_my_acq_func, is_valid_spark_config, sanitize_spark_config
from .acq_optimizer.local_random import InterleavedLocalAndRandomSearch
from LOCAT import QCSA, IICP, DAGP
from LOCAT.dagp_wrapper import DAGPWrapper


class LOCATAdvisor(BaseAdvisor):
    def __init__(self, config_space: ConfigurationSpace, method_id='LOCAT',
                surrogate_type='gp', acq_type='ei', task_id='test',
                ws_strategy='none', ws_args={'init_num': 5},
                tl_strategy='none', tl_args={'topk': 5}, cp_args={},
                random_kwargs={},
                n_qcsa: int = 20,
                n_iicp: int = 20,
                scc_threshold: float = 0.2,
                kpca_kernel: str = 'rbf',
                data_size: float = 100.0,
                **kwargs):
        
        assert tl_strategy == 'none', f"LOCAT requires tl_strategy='none', got '{tl_strategy}'"
        
        super().__init__(config_space, task_id=task_id, method_id=method_id,
                        ws_strategy=ws_strategy, ws_args=ws_args,
                        tl_strategy=tl_strategy, tl_args=tl_args, cp_args=cp_args,
                        **random_kwargs, **kwargs)
        
        self.config_space = config_space
        self.n_qcsa = n_qcsa
        self.n_iicp = n_iicp
        self.scc_threshold = scc_threshold
        self.kpca_kernel = kpca_kernel
        self.data_size = data_size
        self.qcsa: Optional[QCSA] = None
        self.iicp: Optional[IICP] = None
        self.dagp: Optional[DAGP] = None
        
        self.rqa_queries: List[str] = []
        self.compressed_config_space: Optional[ConfigurationSpace] = None
        
        self.qcsa_done = False
        self.iicp_done = False
        self.dagp_initialized = False
        
        self.acq_type = acq_type
        self.surrogate_type = surrogate_type
        self.norm_y = False if 'wrk' in self.acq_type else True
        
        self.surrogate = None
        self.acq_func = None
        self.acq_optimizer = None
    
    def run_qcsa(self, sql_partitioner: Optional[object] = None):
        if self.qcsa_done:
            logger.info("LOCAT: QCSA already completed, skipping...")
            return
        
        logger.info("LOCAT: Starting QCSA analysis using current task history...")
        
        source_history = self.history
        
        if len(source_history) == 0:
            logger.warning("LOCAT: Source history is empty, skipping QCSA")
            return
        
        if sql_partitioner is not None:
            all_queries = sql_partitioner.get_all_sqls()
        else:
            partitioner = self.task_manager.get_sql_partitioner()
            if partitioner is not None:
                all_queries = partitioner.get_all_sqls()
            else:
                logger.warning("LOCAT: Cannot get query list, skipping QCSA")
                return
        
        if not all_queries:
            logger.warning("LOCAT: No queries found, skipping QCSA")
            return
        
        self.qcsa = QCSA(min_samples=max(5, self.n_qcsa))
        
        csq_queries, ciq_queries = self.qcsa.analyze(
            history=source_history,
            all_queries=all_queries
        )
        
        self.rqa_queries = self.qcsa.get_rqa_queries()
        self.qcsa_done = True
        
        qcsa_info = self.qcsa.get_analysis_info()
        self.task_manager.update_history_meta_info({'qcsa': qcsa_info})
        
        self._update_planner_with_rqa()
        for obs in self.history.observations:
            query_times = obs.extra_info['qt_time']
            new_obj = 0
            for q in self.rqa_queries:
                if q in query_times:
                    new_obj += query_times[q]
                else:
                    new_obj = np.inf
                    break
            obs.objectives[0] = new_obj
        
        logger.info(f"LOCAT: QCSA completed. RQA has {len(self.rqa_queries)} queries "
                   f"(removed {len(ciq_queries)} CIQ queries)")
    
    def run_iicp(self, data_size: float = 1.0):
        if self.iicp_done:
            logger.info("LOCAT: IICP already completed, skipping...")
            return
        
        logger.info("LOCAT: Starting IICP analysis using current task history...")
        
        source_history = self.history
        
        if len(source_history) == 0:
            logger.warning("LOCAT: Source history is empty, skipping IICP")
            return
        
        self.iicp = IICP(
            min_samples=max(1, self.n_iicp),
            scc_threshold=self.scc_threshold,
            kpca_kernel=self.kpca_kernel
        )
        
        selected_params, kpca = self.iicp.analyze(
            history=source_history,
            config_space=self.config_space,
            data_size=data_size
        )
        
        self.iicp_done = True
        
        if selected_params:
            self.compressed_config_space = ConfigurationSpace()
            for param_name in selected_params:
                if param_name in self.config_space:
                    self.compressed_config_space.add_hyperparameter(
                        self.config_space[param_name]
                    )
            logger.info(f"LOCAT: IICP created compressed space with {len(selected_params)} parameters")
        else:
            self.compressed_config_space = copy.deepcopy(self.config_space)
            logger.warning("LOCAT: IICP selected no parameters, using original space")
        
        self._compress_history_to_compressed_space()
        
        iicp_info = self.iicp.get_analysis_info()
        self.task_manager.update_history_meta_info({'iicp': iicp_info})
        
        logger.info("LOCAT: IICP completed")
    
    def initialize_dagp(self):
        if self.dagp_initialized:
            logger.info("LOCAT: DAGP already initialized, skipping...")
            return
        
        config_space_for_dagp = self.compressed_config_space or self.sample_space
        
        logger.info("LOCAT: Initializing DAGP...")
        
        iicp_model = self.iicp if self.iicp_done else None
        
        dagp_instance = DAGP(
            config_space=config_space_for_dagp,
            iicp_model=iicp_model,
            kernel=self.surrogate_type if self.surrogate_type.startswith('gp') else 'gp',
            seed=self.seed
        )
        
        if self.iicp_done and self.iicp is not None:
            dagp_instance._init_types_bounds()
        
        self.dagp = dagp_instance
        
        self.surrogate = DAGPWrapper(dagp_instance, config_space_for_dagp)
        
        self.acq_func = build_my_acq_func(func_str=self.acq_type, model=self.surrogate)
        
        self.acq_optimizer = InterleavedLocalAndRandomSearch(
            acquisition_function=self.acq_func,
            rand_prob=self.rand_prob,
            rand_mode=self.rand_mode,
            rng=self.rng,
            config_space=self.compressed_config_space or self.sample_space
        )
        
        self.dagp_initialized = True
        logger.info("LOCAT: DAGP initialized")
    
    def _update_planner_with_rqa(self):
        if not self.rqa_queries:
            logger.warning("LOCAT: No RQA queries to update planner")
            return
        
        partitioner = self.task_manager.get_sql_partitioner()
        if partitioner is None:
            from Evaluator.partitioner import SQLPartitioner
            partitioner = SQLPartitioner(custom_sqls=self.rqa_queries)
            self.task_manager.register_sql_partitioner(partitioner)
            logger.info(f"LOCAT: Created partitioner with {len(self.rqa_queries)} RQA queries")
        else:
            partitioner.set_custom_sqls(self.rqa_queries)
            logger.info(f"LOCAT: Set {len(self.rqa_queries)} RQA queries as custom SQL list")
        
        planner = self.task_manager.get_planner()
        if planner is not None:
            planner.fallback_sqls = {1.0: self.rqa_queries}
            planner._cached_plan = None
            logger.info("LOCAT: Updated planner fallback_sqls and cleared cache")
    
    def warm_start(self):
        pass
    
    def sample(self, batch_size=1, prefix=''):
        num_evaluated_exclude_default = self.get_num_evaluated_exclude_default()
        
        if not self.qcsa_done or not self.iicp_done:
            logger.warning("LOCAT: QCSA or IICP not completed. "
                        "Cannot sample configurations. Please run QCSA and IICP first.")
            return self.sample_random_configs(
                self.sample_space, batch_size,
                excluded_configs=self.history.configurations
            )
        
        if not self.dagp_initialized:
            self.initialize_dagp()
        
        if num_evaluated_exclude_default < self.init_num:
            logger.info(f"LOCAT: Initialization phase ({num_evaluated_exclude_default}/{self.init_num})")
            batch = []
            sample_space = self.compressed_config_space or self.sample_space
            for _ in range(batch_size):
                config = self.sample_random_configs(
                    sample_space, 1,
                    excluded_configs=self.history.configurations + batch
                )[0]
                config.origin = prefix + 'LOCAT Random Init'
                batch.append(config)
            return batch
        
        logger.info("LOCAT: Using DAGP-based acquisition function")
        
        X_config = []
        data_sizes = []
        y = []
        
        for obs in self.history.observations:
            config = obs.config
            objective = obs.objectives[0]
            
            if np.isfinite(objective):
                X_config.append(config)
                data_sizes.append(self.data_size)
                y.append(objective)
        
        if len(X_config) < 2:
            logger.warning("LOCAT: Insufficient training data, using random sampling")
            return self.sample_random_configs(
                self.sample_space, batch_size,
                excluded_configs=self.history.configurations
            )
        
        try:
            self.dagp.train(X_config, np.array(y), data_sizes=np.array(data_sizes))
        except Exception as e:
            logger.error(f"LOCAT: Failed to train DAGP: {e}")
            return self.sample_random_configs(
                self.sample_space, batch_size,
                excluded_configs=self.history.configurations
            )
        
        incumbent_value = self.history.get_incumbent_value()
        self.acq_func.update(
            model=self.surrogate,
            eta=incumbent_value,
            num_data=len(self.history)
        )
        
        observations = self.history.observations
        try:
            challengers = self.acq_optimizer.maximize(
                observations=observations,
                num_points=2000
            )
        except Exception as e:
            logger.error(f"LOCAT: Failed to maximize acquisition function: {e}")
            challengers = type('obj', (object,), {'challengers': []})()
            challengers.challengers = []
        
        batch = []
        for config in challengers.challengers:
            if len(batch) >= batch_size:
                break
            if config in self.history.configurations:
                continue
            
            if not is_valid_spark_config(config):
                config = sanitize_spark_config(config)
            if is_valid_spark_config(config):
                config.origin = prefix + 'LOCAT Acquisition'
                batch.append(config)
        
        if len(batch) < batch_size:
            remaining = batch_size - len(batch)
            sample_space = self.compressed_config_space or self.sample_space
            random_configs = self.sample_random_configs(
                sample_space, remaining,
                excluded_configs=self.history.configurations + batch
            )
            for config in random_configs:
                config.origin = prefix + 'LOCAT Random Fill'
                batch.append(config)
        
        return batch
        
    def _compress_history_to_compressed_space(self):
        if self.compressed_config_space is None:
            return
        
        if self.history is None or len(self.history) == 0:
            return
        
        compressed_names = self.compressed_config_space.get_hyperparameter_names()
        logger.info(f"LOCAT: Compressing {len(self.history.observations)} history observations to compressed space ({len(compressed_names)} params)")
        
        for obs in self.history.observations:
            config = obs.config
            if hasattr(config, 'configuration_space') and config.configuration_space == self.compressed_config_space:
                continue
            
            config_dict = config.get_dictionary()
            filtered_values = {name: config_dict[name] for name in compressed_names if name in config_dict}
            
            new_config = Configuration(self.compressed_config_space, values=filtered_values)
            
            if hasattr(config, 'origin') and config.origin is not None:
                new_config.origin = config.origin
            
            obs.config = new_config
        
        self.history.config_space = self.compressed_config_space
        logger.info("LOCAT: History observations compressed to compressed space")
    