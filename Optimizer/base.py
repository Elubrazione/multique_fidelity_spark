import os
import copy
import numpy as np
import pickle as pkl
from datetime import datetime
from ConfigSpace import ConfigurationSpace
from openbox.utils.history import Observation, History
from openbox import logger
from Advisor.BO import BO
from Advisor.MFBO import MFBO
from .utils import run_obj_func


class BaseOptimizer:
    def __init__(self, config_space: ConfigurationSpace, eval_func, 
                 iter_num=200, per_run_time_limit=None, meta_feature=None,
                 source_hpo_data=None, ep_args=None, ep_strategy='none',
                 method_id='advisor', task_id='test', target='redis', task_str='run',
                 ws_strategy='none', ws_args=None, tl_strategy='none', tl_args=None,
                 cprs_strategy='none', cp_args=None,
                 backup_flag=False, save_dir='./results',
                 seed=42, rand_prob=0.15, rand_mode='ran',
                 config_modifier=None, expert_modified_space=None, expert_params=[],
                 enable_range_compression=False):

        assert method_id in ['RS', 'SMAC', 'GP', 'GPF', 'MFSE_SMAC', 'MFSE_GP', 'BOHB_GP', 'BOHB_SMAC', 'FlexHB_SMAC']
        assert ws_strategy in ['none', 'best_rover', 'rgpe_rover', 'best_all']
        assert tl_strategy in ['none', 'mce', 're', 'mceacq', 'reacq']
        assert ep_strategy in ['none', 'pibo', 'bo_pro', 'prior_band']
        assert cprs_strategy in ['ottertune', 'perrone', 'tuneful', 'locat', 'opadviser', 'rover', 'rover-s', 'rover-l', 'rover-g', 'expert', 'none']
        assert rand_mode in ['ran', 'rs']  # 如果是rs，就是返回random search里面排序过后的，而不是纯随机

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
        if tl_topk > 0:
            tl_str = '%sk%d' % (tl_strategy, tl_topk)
        else:
            tl_str = tl_strategy
            
        cp_topk = cp_args['topk'] if cprs_strategy != 'none' and cp_args['topk'] > 0 \
                                    else len(config_space)
        cp_str = '%sk%dsigma%.1ftop_ratio%.1f' % (cprs_strategy, cp_topk, cp_args['sigma'], cp_args['top_ratio'])

        self.method_id = method_id
        if rand_mode == 'rs':
            self.method_id += 'rs'
        self.task_id = '%s_%s%s__%sW%sT%sE%sC%s__s%d' % (task_id, target, task_str, self.method_id,
                                                         ws_str, tl_str, ep_strategy, cp_str, seed)
        self.target = target
        self.task_str = task_str

        self.ws_strategy = ws_strategy
        self.ws_args = ws_args
        self.tl_strategy = tl_strategy
        self.ep_args = ep_args
        self.ep_strategy = ep_strategy
        self.cprs_strategy = cprs_strategy
        self.cp_args = cp_args

        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.backup_flag = backup_flag
        self.save_dir = save_dir

        self.result_path = None
        self.ts_backup_file = None
        self.ts_recorder = None
        self._logger_kwargs = None  # 传给Advisor，避免rgpe_rover创建新的logger
        self.build_path()
        
        self.config_modifier = config_modifier
        self.expert_modified_space = expert_modified_space

        """
            method_id: SMAC, GP / MFSE_SMAC, MFSE_SMAC, BOHB_GP, BOHB_SMAC, FlexHB_SMAC
            ws_strategy: none, best_rover, rgpe_rover
            tl_strategy: none, mce, re, mceacq, reacq
            expert_strategy: none, pibo, bo_pro, prior_band
            cprs_strategy: none, ottertune, perrone, tuneful, locat, opadviser, rover, rover-s, rover-l, rover-g
        """
        surrogate_type = 'prf'
        if method_id == 'GP':
            surrogate_type = 'gp'
        elif method_id == 'GPF':
            method_id = 'GP'
            surrogate_type = 'gpf'
    
        acq_type = ep_strategy if ep_strategy in ['pibo', 'bo_pro'] else 'ei'
        if tl_strategy != 'none':
            surrogate_type = '%s_%s' % (tl_strategy, surrogate_type)
            if 'acq' in tl_strategy:
                acq_type = 'wrk_%s' % acq_type  # 'wrk_ei'

        if method_id in ['SMAC', 'GP'] or 'BOHB' in method_id:
            self.advisor = BO(config_space, source_hpo_data=source_hpo_data,
                              surrogate_type=surrogate_type, acq_type=acq_type, task_id=self.task_id,
                              ws_strategy=ws_strategy, ws_args=ws_args, tl_args=tl_args,
                              ep_args=ep_args, ep_strategy=ep_strategy, meta_feature=meta_feature,
                              cprs_strategy=cprs_strategy, cp_args=cp_args,
                              safe_flag=False, seed=seed, rng=self.rng, rand_prob=rand_prob, rand_mode=rand_mode,
                              expert_modified_space=expert_modified_space, expert_params=expert_params,
                              _logger_kwargs=self._logger_kwargs)
        elif 'MFSE' in method_id or 'FlexHB' in method_id:
            # 对于MFSE，没有tl，就默认用MF集成
            if tl_strategy == 'none':
                surrogate_type = 'mfse_' + surrogate_type

            self.advisor = MFBO(config_space, source_hpo_data=source_hpo_data, meta_feature=meta_feature,
                                surrogate_type=surrogate_type, acq_type=acq_type, task_id=self.task_id,
                                ws_strategy=ws_strategy, ws_args=ws_args, tl_args=tl_args,
                                cprs_strategy=cprs_strategy, cp_args=cp_args,
                                safe_flag=False, seed=seed, rng=self.rng, rand_prob=rand_prob, rand_mode=rand_mode,
                                expert_params=expert_params,
                                enable_range_compression=enable_range_compression,
                                _logger_kwargs=self._logger_kwargs)

        self.timeout = per_run_time_limit

    def build_path(self):

        # 结果保存
        # results
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

        _logger_kwargs = {'name': "%s" % self.task_id, 'logdir': './log/%s/%s' % (self.target, self.method_id)}
        logger.init(**_logger_kwargs)
        _logger_kwargs['force_init'] = False
        self._logger_kwargs = _logger_kwargs

    def run(self):
        while self.iter_id < self.iter_num:
            self.run_one_iter()

    """
    将任务的关键信息保存
    """
    def record_task(self):
        if self.iter_id >= 25:
            self.ts_recorder.append(copy.deepcopy(self.advisor.history))
            logger.warn("[{}] Successfully record task!".format(self.task_str))
        else:
            logger.warn("[{}] Failed to record the task because the number of iterations was less than 25!".format(
                self.task_str))

    def run_one_iter(self):

        self.iter_id += 1

        config = self.advisor.sample()
        if self.config_modifier is not None:
            config = self.config_modifier(config)
            
        logger.info(
            "[{}] iter ------------------------------------------------{:5d}".format(self.task_str, self.iter_id))
        if config.origin:
            logger.warn("[{}] !!!!!!!!!! {} !!!!!!!!!!".format(self.task_str, config.origin))

        obj_args, obj_kwargs = (config,), dict() # 这里只是将参数包装了一下, run_obj_func里面又会将他们解开, 解开后本质上是

        results = run_obj_func(self.eval_func, obj_args, obj_kwargs, self.timeout) # 返回一个字典, 其中`results`中的内容就是上面的{perf:...}
        self.advisor.update(config, results)

        logger.info('[{}] Config: '.format(self.task_str) + str(config.get_dictionary()))
        logger.info('[{}] Obj: {}, best obj: {}'.format(self.task_str, results['result']['objective'], self.advisor.history.get_incumbent_value()))

        logger.info(
            "[{}] ===================================================================================================================================================".format(
                self.task_str))

        self.save_info()

    def save_info(self, interval=1):
        # 将迁移学习的w保存
        if self.tl_strategy != 'none' or 'MFSE' in self.method_id:
            hist_ws = self.advisor.surrogate.hist_ws.copy()
            self.advisor.history.meta_info['tl_ws'] = hist_ws
        
        if self.iter_id == self.iter_num or self.iter_id % interval == 0:
            # his = self.advisor.history
            his = copy.deepcopy(self.advisor.history)
            
            if hasattr(self.advisor, 'fixed_indices') and self.advisor.fixed_indices:
                param_names = [param.name for param in self.advisor.config_space.get_hyperparameters()]
                data = {
                    'task_id': his.task_id,
                    'num_objectives': his.num_objectives,
                    'num_constraints': his.num_constraints,
                    'ref_point': his.ref_point,
                    'meta_info': his.meta_info,
                    'global_start_time': his.global_start_time.isoformat(),
                    'observations': [
                        obs.to_dict() for obs in his.observations
                    ]
                }

                for obs in data['observations']:
                    new_conf = {}
                    for name in param_names:
                        if name in obs['config']:
                            new_conf[name] = obs['config'][name]
                    obs['config'] = new_conf
                    
                global_start_time = data.pop('global_start_time')
                global_start_time = datetime.fromisoformat(global_start_time)
                observations = data.pop('observations')
                observations = [Observation.from_dict(obs, self.advisor.config_space) for obs in observations]

                his = History(**data)
                his.global_start_time = global_start_time
                his.update_observations(observations)
                
            his.save_json(self.result_path)
            
            if self.iter_id == self.iter_num and self.backup_flag:
                self.record_task()
                with open(self.ts_backup_file, 'wb') as ts:
                    pkl.dump(self.ts_recorder, ts)

