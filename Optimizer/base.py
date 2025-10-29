import os
import copy
import numpy as np
import pickle as pkl
from datetime import datetime
from ConfigSpace import ConfigurationSpace
from openbox import logger
from Advisor.BO import BO
from Advisor.MFBO import MFBO
from .utils import run_obj_func
from Advisor.task_manager import TaskManager


class BaseOptimizer:
    def __init__(self, config_space: ConfigurationSpace, eval_func, task_manager: TaskManager,
                 iter_num=200, per_run_time_limit=None,
                 method_id='advisor', task_id='test', target='redis',
                 ws_strategy='none', ws_args=None, tl_strategy='none', tl_args=None,
                 cprs_strategy='none', cp_args=None,
                 backup_flag=False, save_dir='./results',
                 seed=42, rand_prob=0.15, rand_mode='ran', _logger_kwargs=None):

        assert method_id in ['RS', 'SMAC', 'GP', 'GPF', 'MFSE_SMAC', 'MFSE_GP', 'BOHB_GP', 'BOHB_SMAC']
        assert ws_strategy in ['none', 'best_rover', 'rgpe_rover', 'best_all']
        assert tl_strategy in ['none', 'mce', 're', 'mceacq', 'reacq']
        assert cprs_strategy in ['shap', 'expert', 'none']
        assert rand_mode in ['ran', 'rs']

        self.eval_func = eval_func
        self.iter_num = iter_num
        self.iter_id = 0

        ws_str = ws_strategy
        if method_id != 'RS':
            init_num = ws_args['init_num']
            if 'rgpe' not in ws_strategy:
                ws_str = '%s%d' % (ws_strategy, init_num)
            else:
                ws_topk = ws_args['topk']
                ws_str = '%s%dk%d' % (ws_strategy, init_num, ws_topk)

        tl_topk = tl_args['topk'] if tl_strategy != 'none' else -1
        tl_str = '%sk%d' % (tl_strategy, tl_topk)            
        cp_topk = cp_args['topk'] if cprs_strategy != 'none' and cp_args['topk'] > 0 \
                                    else len(config_space)
        cp_str = '%sk%dsigma%.1ftop_ratio%.1f' % (cprs_strategy, cp_topk, cp_args['sigma'], cp_args['top_ratio'])

        self.method_id = method_id
        if rand_mode == 'rs':
            self.method_id += 'rs'
        self.task_id = '%s__%s__W%sT%sC%s__s%d' % (task_id, self.method_id,
                                                        ws_str, tl_str, cp_str, seed)
        self.target = target

        self.ws_strategy = ws_strategy
        self.ws_args = ws_args
        self.tl_strategy = tl_strategy
        self.cprs_strategy = cprs_strategy
        self.cp_args = cp_args

        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.backup_flag = backup_flag
        self.save_dir = save_dir

        self.result_path = None
        self.ts_backup_file = None
        self.ts_recorder = None
        self._logger_kwargs = _logger_kwargs

        self.build_path()
        

        """
            method_id: SMAC, GP / MFSE_SMAC, MFSE_SMAC, BOHB_GP, BOHB_SMAC
            ws_strategy: none, best_rover, rgpe_rover
            tl_strategy: none, mce, re, mceacq, reacq
            cprs_strategy: none, shap
        """
        surrogate_type = 'prf'
        if method_id == 'GP':
            surrogate_type = 'gp'
        elif method_id == 'GPF':
            method_id = 'GP'
            surrogate_type = 'gpf'
    
        acq_type = 'ei'
        if tl_strategy != 'none':
            surrogate_type = '%s_%s' % (tl_strategy, surrogate_type)
            if 'acq' in tl_strategy:
                acq_type = 'wrk_%s' % acq_type  # 'wrk_ei'

        if method_id in ['SMAC', 'GP'] or 'BOHB' in method_id:
            self.advisor = BO(config_space,
                              surrogate_type=surrogate_type, acq_type=acq_type, task_id=self.task_id,
                              ws_strategy=ws_strategy, ws_args=ws_args, tl_args=tl_args,
                              cprs_strategy=cprs_strategy, cp_args=cp_args,
                              seed=seed, rng=self.rng, rand_prob=rand_prob, rand_mode=rand_mode,
                              task_manager=task_manager,
                              _logger_kwargs=self._logger_kwargs)
        elif 'MFSE' in method_id:
            # 对于MFSE，没有tl，就默认用MF集成
            if tl_strategy == 'none':
                surrogate_type = 'mfse_' + surrogate_type

            self.advisor = MFBO(config_space,
                                surrogate_type=surrogate_type, acq_type=acq_type, task_id=self.task_id,
                                ws_strategy=ws_strategy, ws_args=ws_args, tl_args=tl_args,
                                cprs_strategy=cprs_strategy, cp_args=cp_args,
                                seed=seed, rng=self.rng, rand_prob=rand_prob, rand_mode=rand_mode,
                                task_manager=task_manager,
                                _logger_kwargs=self._logger_kwargs)

        self.timeout = per_run_time_limit

    def build_path(self):
        result_dir = os.path.join(self.save_dir, self.target, self.method_id)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        self.res_dir = os.path.join(result_dir, timestamp)
        os.makedirs(self.res_dir)
        
        self.result_path = os.path.join(self.res_dir, "%s_%s.json" % (self.task_id, timestamp))

        # 过去的任务的备份
        self.ts_backup_file = "./backup/ts_backup_%s.pkl" % self.target
        if not os.path.exists("./backup"):
            os.makedirs("./backup")
        # log
        try:
            self.ts_recorder = pkl.load(open(self.ts_backup_file, 'rb'))
            logger.warn("Successfully initialize from %s !" % self.ts_backup_file)
        except FileNotFoundError:
            self.ts_recorder = []
            logger.warn("File \"%s\" not found, initialize to empty list" % self.ts_backup_file)


    def run(self):
        while self.iter_id < self.iter_num:
            self.run_one_iter()

    """
    将任务的关键信息保存
    """
    def record_task(self):
        if self.iter_id >= 25:
            self.ts_recorder.append(copy.deepcopy(self.advisor.history))
            logger.warn("Successfully record task!")
        else:
            logger.warn("Failed to record the task because the number of iterations was less than 25!")

    def run_one_iter(self):

        self.iter_id += 1

        config = self.advisor.sample()
        obj_args, obj_kwargs = (config,), dict() # 这里只是将参数包装了一下, run_obj_func里面又会将他们解开, 解开后本质上是
        results = run_obj_func(self.eval_func, obj_args, obj_kwargs, self.timeout)
        self.advisor.update(config, results)
        self.log_iteration_results([config], [results['result']['objective']])
        self.save_info()

    def log_iteration_results(self, configs, performances, iteration_id=None):
        iter_id = iteration_id if iteration_id is not None else self.iter_id
        
        logger.info("[{}] iter ------------------------------------------------ {:5d}".format(iter_id))
        
        for idx, config in enumerate(configs):
            if hasattr(config, 'origin') and config.origin:
                logger.warn("!!!!!!!!!! {} !!!!!!!!!!".format(config.origin))
            
            logger.info('Config: ' + str(config.get_dictionary()))
            logger.info('Obj: {}'.format(performances[idx]))
        
        logger.info('best obj: {}'.format(self.advisor.history.get_incumbent_value()))
        logger.info("===================================================================================================================================================")

    def save_info(self, interval=1):
        if self.tl_strategy != 'none' or 'MFSE' in self.method_id:
            hist_ws = self.advisor.surrogate.hist_ws.copy()
            self.advisor.history.meta_info['tl_ws'] = hist_ws
        
        if self.iter_id == self.iter_num or self.iter_id % interval == 0:    
            self.advisor.history.save_json(self.result_path)
            
            if self.iter_id == self.iter_num and self.backup_flag:
                self.record_task()
                with open(self.ts_backup_file, 'wb') as ts:
                    pkl.dump(self.ts_recorder, ts)

