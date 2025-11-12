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
from .utils import run_obj_func
from .scheduler import schedulers
from Advisor import advisors
from task_manager import TaskManager


class TunefulOptimizer:
    def __init__(self, config_space: ConfigurationSpace, eval_func,
                 iter_num=200, per_run_time_limit=None,
                 method_id='advisor', task_id='test',
                 target='redis', save_dir='./results',
                 ws_strategy='none', tl_strategy='none',
                 backup_flag=False, resume: Optional[str] = None):

        assert method_id in ['tuneful']

        scheduler_type = 'full'
        
        self.eval_func = eval_func
        self.iter_num = iter_num

        task_mgr = TaskManager.instance()
        self.scheduler_kwargs = task_mgr.get_scheduler_kwargs()
        self.scheduler = schedulers[scheduler_type](
            num_nodes=len(task_mgr._config_manager.multi_nodes),
            **self.scheduler_kwargs
        )
        task_mgr.register_scheduler(self.scheduler)

        self.random_kwargs = task_mgr.get_random_kwargs()
        self._logger_kwargs = task_mgr.get_logger_kwargs()
        self.iter_id = len(task_mgr.current_task_history) - 1 if resume is not None else 0
        

        self.method_id = method_id
        self.task_id = '%s__%s__S%s__s%d' % (task_id, self.method_id, scheduler_type, self.random_kwargs.get('seed', 42))

        self.save_dir = save_dir
        self.target = target
        self.result_path = None

        self.build_path()
        
        advisor_class = advisors['tuneful']
        self.advisor = advisor_class(
            config_space=config_space,
            _logger_kwargs = self._logger_kwargs
        )

        self.timeout = per_run_time_limit
        self.save_info()


    def build_path(self):
        self.res_dir = os.path.join(self.save_dir, self.target)
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir)

        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')       
        self.result_path = os.path.join(self.res_dir, "%s_%s.json" % (self.task_id, timestamp))


    def run(self):
        while self.iter_id < self.iter_num:
            self.run_one_iter()


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
            self.iter_id
        )

        for i in range(s + 1):
            n_configs, n_resource = self.scheduler.get_stage_params(s=s, stage=i)
            logger.info(f"Stage {i}: n_configs={n_configs}, n_resource={n_resource}")
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

        candidates, perfs = self._iterate()

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
        
        if self.iter_id == self.iter_num or self.iter_id % interval == 0:    
            self._save_json_atomic(self.result_path)
            
    
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

class RoverOptimizer:
    def __init__(self, config_space: ConfigurationSpace, eval_func, 
                 iter_num=200, per_run_time_limit=None,
                 method_id='advisor', task_id='test', 
                 target='redis', save_dir='./results', 
                 ws_strategy='none', tl_strategy='none',
                 backup_flag=False, resume: Optional[str] = None):

        assert method_id in ['rover']

        scheduler_type = 'full'

        self.eval_func = eval_func
        self.iter_num = iter_num

        task_mgr = TaskManager.instance()
        self.scheduler_kwargs = task_mgr.get_scheduler_kwargs()
        self.scheduler = schedulers[scheduler_type](
            num_nodes=len(task_mgr._config_manager.multi_nodes),
            **self.scheduler_kwargs
        )
        task_mgr.register_scheduler(self.scheduler)

        self.random_kwargs = task_mgr.get_random_kwargs()
        self._logger_kwargs = task_mgr.get_logger_kwargs()
        self.iter_id = len(task_mgr.current_task_history) - 1 if resume is not None else 0

        self.method_id = method_id
        self.task_id = '%s__%s__S%s__s%d' % (task_id, self.method_id, scheduler_type, self.random_kwargs.get('seed', 42))

        self.save_dir = save_dir
        self.target = target
        self.result_path = None

        self.build_path()

        advisor_class = advisors['rover']
        self.advisor = advisor_class(
            config_space=config_space,
            _logger_kwargs = self._logger_kwargs
        )

        self.timeout = per_run_time_limit
        self.save_info()

    def build_path(self):
        result_dir = os.path.join(self.save_dir, self.target, self.method_id)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        self.result_path = os.path.join(result_dir, "%s_%s.json" % (self.task_id, timestamp))

    def run(self):
        while self.iter_id < self.iter_num:
            self.run_one_iter()

    def _evaluate_configurations( # 暂时猜测此函数的作用是替代原本直接调用eval_func的地方, 以便并行评估多个config (同时适配多精度调优)
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
            self.iter_id
        )

        for i in range(s + 1):
            n_configs, n_resource = self.scheduler.get_stage_params(s=s, stage=i)
            logger.info(f"Stage {i}: n_configs={n_configs}, n_resource={n_resource}")
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

        candidates, perfs = self._iterate() # 原本advisor的sample也被自动移动到_iterate中, 并行评估也被移动到_iterate中
        
        self.log_iteration_results(candidates, perfs)
        self.save_info()

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

    def save_info(self, interval=2):

        if self.iter_id == self.iter_num or self.iter_id % interval == 0:    
            self._save_json_atomic(self.result_path)

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

