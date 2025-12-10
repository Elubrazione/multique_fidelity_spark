import os
import copy
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pickle as pkl
from datetime import datetime
from ConfigSpace import ConfigurationSpace
from openbox import logger
from typing import List, Optional
from .scheduler import schedulers
from Advisor import get_advisor_config, get_advisor
from manager import TaskManager


class BaseOptimizer:
    def __init__(self, config_space: ConfigurationSpace, eval_func,
                 iter_num=200, per_run_time_limit=None,
                 method_id='advisor', task_id='test',
                 target='redis', save_dir='./results',
                 ws_strategy='none', tl_strategy='none',
                 backup_flag=False, resume: Optional[str] = None):

        # assert method_id in ['RS', 'SMAC', 'GP', 'GPF', 'MFES_SMAC', 'MFES_GP', 'BOHB_GP', 'BOHB_SMAC',
        #                      'LLAMATUNE_SMAC', 'LLAMATUNE_GP', 'REMBO_SMAC', 'REMBO_GP', 'HESBO_SMAC', 'HESBO_GP']
        # assert ws_strategy in ['none', 'best_cos', 'best_euc', 'best_rover', 'rgpe_rover', 'best_all']
        # assert tl_strategy in ['none', 'mce', 're', 'mceacq', 'reacq']
        self.eval_func = eval_func
        self.iter_num = iter_num

        task_mgr = TaskManager.instance()
        self.scheduler_kwargs = task_mgr.get_scheduler_kwargs()
        
        use_flatten = task_mgr._config_manager.args.use_flatten_scheduler
        scheduler_type = 'mfes' if 'MFES' in method_id else 'bohb' if 'BOHB' in method_id else 'full'
        if use_flatten and scheduler_type in ['mfes', 'bohb']:
            scheduler_type = f"{scheduler_type}_flatten"
            logger.info(f"Using flatten scheduler variant: {scheduler_type}")
        self.scheduler = schedulers[scheduler_type](
            num_nodes=len(task_mgr._config_manager.multi_nodes),
            **self.scheduler_kwargs
        )
        task_mgr.register_scheduler(self.scheduler)

        self.random_kwargs = task_mgr.get_random_kwargs()
        self._logger_kwargs = task_mgr.get_logger_kwargs()
        self.iter_id = len(task_mgr.current_task_history) - 1 if resume is not None else 0
        
        self.method_id = method_id
        self.task_id = task_mgr.generate_task_id(
            task_name=task_id,
            method_id=method_id,
            ws_strategy=ws_strategy,
            tl_strategy=tl_strategy,
            scheduler_type=scheduler_type,
            config_space=config_space,
            rand_mode=self.random_kwargs.get('rand_mode', 'ran'),
            seed=self.random_kwargs.get('seed', 42)
        )

        self.ws_strategy = ws_strategy
        self.tl_strategy = tl_strategy

        self.backup_flag = backup_flag
        self.save_dir = save_dir
        self.target = target
        self.result_path = None
        self.ts_backup_file = None
        self.ts_recorder = None

        self.build_path()
        
        advisor_config = get_advisor_config(method_id, tl_strategy)
        method_id = 'GP' if method_id == 'GPF' else method_id
        advisor_class = get_advisor(advisor_config.advisor_type)
        self.advisor = advisor_class(
            config_space=config_space,
            task_id=self.task_id,
            ws_strategy=ws_strategy,
            tl_strategy=tl_strategy,
            _logger_kwargs=self._logger_kwargs,
            method_id=method_id,
            **advisor_config.to_dict(),
            **self.random_kwargs
        )

        self.timeout = per_run_time_limit
        self.save_info()


    def build_path(self):
        self.res_dir = os.path.join(self.save_dir, self.target, self.method_id)
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir)

        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')       
        self.result_path = os.path.join(self.res_dir, "%s_%s.json" % (self.task_id, timestamp))

        self.ts_backup_file = "./backup/ts_backup_%s.pkl" % self.target
        if not os.path.exists("./backup"):
            os.makedirs("./backup")
        try:
            self.ts_recorder = pkl.load(open(self.ts_backup_file, 'rb'))
            logger.warn("Successfully initialize from %s !" % self.ts_backup_file)
        except FileNotFoundError:
            self.ts_recorder = []
            logger.warn("File \"%s\" not found, initialize to empty list" % self.ts_backup_file)


    def run(self):
        while self.iter_id < self.iter_num:
            self.run_one_iter()


    def record_task(self):
        if self.iter_id >= 25:
            self.ts_recorder.append(copy.deepcopy(self.advisor.history))
            logger.warn("Successfully record task!")
        else:
            logger.warn("Failed to record the task because the number of iterations was less than 25!")

    def _evaluate_configurations(
        self, candidates,
        resource_ratio=round(float(1.0), 5)
    ) -> List[float]:

        futures, performances = [], []
        with ThreadPoolExecutor(max_workers=len(candidates)) as executor:
            for config in candidates:
                future = executor.submit(self.eval_func, config=config, resource_ratio=resource_ratio)
                futures.append((future, config))

            for future, config in futures:
                results = future.result()
                self.advisor.update(
                    config=config, results=results,
                    resource_ratio=resource_ratio,
                    update=self.scheduler.should_update_history(resource_ratio)
                )
                performances.append(results['result']['objective'])
        return performances
    
    def _iterate(self):   
        iter_full_eval_configs, iter_full_eval_perfs = [], []
        candidates = []

        s = self.scheduler.get_bracket_index(
            self.iter_id - self.advisor.init_num - self.advisor.has_default_config
        )

        for i in range(s + 1):
            n_configs, n_resource = self.scheduler.get_stage_params(s=s, stage=i)
            logger.info(f"Bracket {s} Stage {i}: n_configs={n_configs}, n_resource={n_resource}")
            if not i:
                candidates = self.advisor.sample(batch_size=n_configs)
                logger.info(f"Generated {len(candidates)} initial candidates")
            resource_ratio = self.scheduler.calculate_resource_ratio(n_resource=n_resource)
            perfs = self._evaluate_configurations(candidates, resource_ratio)            
            candidates, perfs = self.scheduler.eliminate_candidates(candidates, perfs, s=s, stage=i)
            
            if i == s:
                iter_full_eval_configs.extend(candidates)
                iter_full_eval_perfs.extend(perfs)

        return iter_full_eval_configs, iter_full_eval_perfs

    def run_one_iter(self):
        self.iter_id += 1
        logger.info("iter =========================================================================== {:3d}".format(self.iter_id))
        num_config_evaluated = self.advisor.get_num_evaluated_exclude_default()
        if num_config_evaluated < self.advisor.init_num:
            candidates = self.advisor.sample(batch_size=self.scheduler.num_nodes)
            logger.info(f"Initialization phase: need to evaluate {self.scheduler.num_nodes} configs, generated {len(candidates)} initial candidates")
            perfs = self._evaluate_configurations(candidates, resource_ratio=round(float(1.0), 5))
        else:
            candidates, perfs = self._iterate()
        
        if hasattr(self.advisor, 'update_compression'):
            self.advisor.update_compression(self.advisor.history)

        self.log_iteration_results(candidates, perfs)
        self.save_info(interval=1)

    def log_iteration_results(self, configs, performances):
        logger.info("------------------------------------------------------------------------------")
        for idx, config in enumerate(configs):
            if hasattr(config, 'origin') and config.origin:
                logger.warn("!!!!!!!!!! {} !!!!!!!!!!".format(config.origin))
            
            logger.info('Config: ' + str(config.get_dictionary()))
            logger.info('Obj: {}'.format(performances[idx]))
            logger.info("-------------------------------------------------------------------------------")

        logger.info('best obj: {}'.format(self.advisor.history.get_incumbent_value()))
        logger.info("===============================================================================")

    def save_info(self, interval=1):
        if self.tl_strategy != 'none' or 'MFES' in self.method_id:
            hist_ws = self.advisor.surrogate.hist_ws.copy()
            self.advisor.history.meta_info['tl_ws'] = hist_ws
        
        if self.iter_id == self.iter_num or self.iter_id % interval == 0:    
            self._save_json_atomic(self.result_path)
            
            if self.iter_id == self.iter_num and self.backup_flag:
                self.record_task()
                self._save_pkl_atomic(self.ts_backup_file, self.ts_recorder)
    
    def _save_json_atomic(self, filepath: str):
        try:
            temp_dir = os.path.dirname(filepath) or '.'
            temp_fd, temp_path = tempfile.mkstemp(
                suffix='.json',
                dir=temp_dir,
                prefix=os.path.basename(filepath) + '.tmp.'
            )
            try:
                os.close(temp_fd)
                self.advisor.history.save_json(temp_path)
                shutil.move(temp_path, filepath)
                logger.debug(f"Successfully saved history to {filepath}")
            except Exception as e:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                raise e
        except Exception as e:
            logger.error(f"Failed to save history to {filepath}: {e}")
            raise
    
    def _save_pkl_atomic(self, filepath: str, data):
        try:
            temp_dir = os.path.dirname(filepath) or '.'
            temp_fd, temp_path = tempfile.mkstemp(
                suffix='.pkl',
                dir=temp_dir,
                prefix=os.path.basename(filepath) + '.tmp.'
            )
            
            try:
                with open(temp_path, 'wb') as f:
                    pkl.dump(data, f)
                
                shutil.move(temp_path, filepath)
                logger.debug(f"Successfully saved backup to {filepath}")
            except Exception as e:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                raise e
        except Exception as e:
            logger.error(f"Failed to save backup to {filepath}: {e}")
            raise