import json as js
import numpy as np
import copy
from datetime import datetime
from itertools import combinations
from openbox import logger, space as sp
from openbox.utils.history import Observation, History
from openbox.utils.config_space.util import convert_configurations_to_array
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.read_and_write.json import write

from .base import BaseAdvisor
from .utils import build_my_surrogate, build_my_acq_func
from .workload_mapping.rover.transfer import get_transfer_suggestion
from .acq_optimizer.local_random import InterleavedLocalAndRandomSearch
from Compressor.utils import build_my_compressor


class BO(BaseAdvisor):
    def __init__(self, config_space: ConfigurationSpace, source_hpo_data=None,
                 surrogate_type='prf', acq_type='ei', task_id='test', meta_feature=None,
                 ws_strategy='none', ws_args={'init_num': 5}, tl_args={'topk': 5},
                 ep_args=None, ep_strategy='none', expert_params=[],
                 cp_topk=50, space_history = None, cprs_strategy='none',
                 safe_flag=False, seed=42, rng=None, rand_prob=0.15, rand_mode='ran', 
                 expert_modified_space=None, enable_range_compression=False,
                 range_compress_data_path=None, **kwargs):
        super().__init__(config_space, task_id=task_id, meta_feature=meta_feature,
                         ws_strategy=ws_strategy, ws_args=ws_args,
                         tl_args=tl_args, source_hpo_data=source_hpo_data,
                         ep_args=ep_args, ep_strategy=ep_strategy,
                         cprs_strategy=cprs_strategy, space_history=space_history, cp_topk=cp_topk,
                         seed=seed, rng=rng, rand_prob=rand_prob, rand_mode=rand_mode, **kwargs)

        self.safe_flag = safe_flag

        self.acq_type = acq_type
        self.surrogate_type = surrogate_type
        self.extra_dim = 0
        
        self.origin_expert_space = expert_modified_space
        self.expert_modified_space = copy.deepcopy(self.origin_expert_space)
        self.expert_params = expert_params

        self.norm_y = True
        if 'wrk' in acq_type:
            self.norm_y = False

        self.init_num = ws_args['init_num']
        
        # 处理范围压缩
        if enable_range_compression:
            logger.info("启用范围压缩功能，将自动计算压缩空间")
            self.range_config_space = self.compute_range_compression(
                range_compress_data_path=range_compress_data_path,
            )
        else:
            logger.info("未启用范围压缩功能，使用原始搜索空间")
        
        # History 里面存的config是完整的原始空间配置
        # 取 observation 出来的时候需要根据 indices 进行变换得到压缩后的配置
        # Challenger 得到的配置则是压缩后的，需要再把它拼回完整的
        self.compress_space_optimizer(space_history)

    def warm_start(self):
        if self.ws_strategy == 'none':
            return

        sims = self.source_hpo_data_sims

        for i, sim in enumerate(sims):
            logger.info("The %d-th similar task(%s): %s" % (sim[0], self.source_hpo_data[i].task_id, sim[1]))

        warm_str_list = []
        for i in range(len(sims)):
            idx, sim = sims[i]
            task_str = self.source_hpo_data[idx].task_id
            warm_str = "%s: sim%.4f" % (task_str, sim)
            warm_str_list.append(warm_str)

        if 'warm_start' not in self.history.meta_info:
            self.history.meta_info['warm_start'] = [warm_str_list]
        else:
            self.history.meta_info['warm_start'].append(warm_str_list)

        num_evaluated = len(self.history)
        if self.ws_strategy.startswith('best'):
            for i, sim in enumerate(sims):
                sim_obs = copy.deepcopy(self.source_hpo_data[sim[0]].observations)
                sim_obs = sorted(sim_obs, key=lambda x: x.objectives[0])

                task_num = 3 if i == 0 else 1
                for j in range(task_num):
                    config_warm_old = sim_obs[j].config
                    # 在 sample_space 里创建新 config，并逐个拷贝参数
                    # 注意这里的搜索空间是 config_space 而不是 sample_space
                    config_warm = Configuration(self.config_space, values={
                        name: config_warm_old[name] for name in self.sample_space.get_hyperparameter_names()
                    })
                    config_warm.origin = self.ws_strategy + self.source_hpo_data[sim[0]].task_id
                    # 后加的更差，因为是从后往前取的，所以往前加
                    self.ini_configs = [config_warm] + self.ini_configs
                if len(self.ini_configs) + num_evaluated >= self.init_num:
                    break

            while len(self.ini_configs) + num_evaluated < self.init_num:
                config = self.sample_random_configs(self.sample_space, 1,
                                                    excluded_configs=self.history.configurations)[0]
                self.ini_configs = [config] + self.ini_configs

            logger.info("Successfully warm start %d configurations with %s!" % (len(self.ini_configs), self.ws_strategy))

        elif self.ws_strategy.startswith('rgpe'):
            topk = self.ws_args.get('topk', 3)

            src_history = [self.source_hpo_data[sims[i][0]] for i in range(topk)]
            target_history = self.history

            while len(self.ini_configs) + num_evaluated < self.init_num:
                final_config = get_transfer_suggestion(src_history, target_history, _logger_kwargs=self._logger_kwargs)
                final_config.origin = self.ws_strategy
                if final_config not in self.history.configurations + self.ini_configs:
                    self.ini_configs.append(final_config)

            logger.info("Successfully warm start %d configurations with %s!" % (len(self.ini_configs), self.ws_strategy))

        else:

            raise ValueError('Invalid ws_strategy: %s' % self.ws_strategy)

    """
    采样(使用ini_configs进行热启动和普通采样)
    以及安全约束 (40轮后阈值为 0.85 * incumbent_value)
    """
    def sample(self, return_list=False):
        num_config_evaluated = len(self.history)
        if len(self.ini_configs) == 0 and (
            (self.init_num > 0 and num_config_evaluated < self.init_num)
            or (self.init_num == 0 and num_config_evaluated == 0)
        ):
            logger.info("Begin to warm start!")
            self.warm_start()

        logger.info("num_config_evaluated: [%d], init_num: [%d], init_configs: [%d]" % (num_config_evaluated, self.init_num, len(self.ini_configs)))
        if num_config_evaluated < self.init_num or (not self.init_num and not num_config_evaluated):
        # if num_config_evaluated <= self.init_num:
            if len(self.ini_configs) > 0:
                config = self.ini_configs[-1]
                self.ini_configs.pop()
            else:
                config = self.sample_random_configs(self.sample_space, 1,
                                                    excluded_configs=self.history.configurations)[0]
            if return_list:
                return [config]
            else:
                return config
    
        X = self.history.get_config_array()
        Y = self.history.get_objectives()
        
        if self.ep_strategy != 'none':
            assert self.ep_strategy in ['pibo', 'bo_pro', 'prior_band']
            '''
            1. 提取历史观察值 X, Y
            2. 计算 prior 预测值 pred_y
            3. 统计 prior 预测排序的准确度，计算 prior_weight
            4. 更新采集函数（） 或 通过随机概率 p 控制配置采样方式
            '''
            prior_func = self.ep_args.get('prior', None)
            assert prior_func is not None
            
            pred_y = np.array([prior_func(x) for x in X])
            pairwise_comparisons = list(combinations(range(num_config_evaluated), 2))
            correct_predictions = sum(
                (pred_y[i] > pred_y[j]) == (Y[i] < Y[j]) for i, j in pairwise_comparisons
            )
            self.prior_weight = correct_predictions / len(pairwise_comparisons)
            
            if self.ep_strategy in ['pibo', 'bo_pro']:
                self.acq_func.update(prior_weight=self.prior_weight)
            elif self.ep_strategy == 'prior_band':  # 直接返回
                if np.random.rand() < self.rand_prob * 2 / 3:
                    return self.sample_random_configs(self.sample_space, 1,
                                                      excluded_configs=self.history.configurations)[0]
                # (prior_weight / (1 + prior_weight)) 概率基于 prior 选择
                if np.random.rand() < self.prior_weight / (1 + self.prior_weight):
                    candidates = self.sample_random_configs(self.sample_space, 2000,
                                                            excluded_configs=self.history.configurations)
                    X = convert_configurations_to_array(candidates)
                    weights = np.array(prior_func(X))
                    # 选择最大权重的配置
                    return candidates[np.argmax(weights)]

        if self.surrogate_type == 'gpf':
            self.surrogate = build_my_surrogate(func_str=self.surrogate_type, config_space=self.config_space,
                                                rng=self.rng,
                                                transfer_learning_history=self.source_hpo_data,
                                                extra_dim=self.extra_dim, norm_y=self.norm_y)
            logger.info("Successfully rebuild the surrogate model GP!")
            
        self.surrogate.train(X, Y)

        incumbent_value = np.sort(Y)[(num_config_evaluated - 1) // 5] \
            if self.ep_strategy == 'bo_pro' else self.history.get_incumbent_value()
        self.acq_func.update(model=self.surrogate,
                             eta=incumbent_value,
                             num_data=num_config_evaluated)

        observations = self.history.observations
        challengers = self.acq_optimizer.maximize(observations=observations, num_points=2000)
        
        if self.cprs_strategy == 'op_advisor':
            pred_y = self.compressor.predict(challengers.get_challenger_array())
            challengers.challengers = [
                challengers.challengers[i] 
                for i in range(len(challengers.challengers)) 
                if pred_y[i] < 1.2 * self.history.get_incumbent_value()
            ] or challengers.challengers
    
        if return_list:
            return challengers.challengers

        cur_config = challengers.challengers[0]
        

        # 对1个任务，在40轮之后引入安全约束（阈值85%)
        if self.safe_flag:
            recommend_flag = True
            if len(self.history) >= 10:
                recommend_flag = False
                for config in challengers.challengers:
                    X = convert_configurations_to_array([config])
                    pred_mean, _ = self.surrogate.predict(X)
                    if pred_mean[0] < 0.85 * self.history.get_incumbent_value():  # 满足约束 (perf是负数)
                        logger.warn(
                            '-----------The config_%d meet the security constraint-----------' % challengers.challengers.index(
                                config))
                        cur_config = config
                        recommend_flag = True
                        break

            if recommend_flag:
                logger.warn("Successfully recommend a configuration through Advisor!")
            else:
                logger.error(
                    "Failed to recommend a configuration that meets the security constraint! Return the incumbent_config")
                cur_config = self.history.get_incumbent_configs()[0]

        else:
            recommend_flag = False
            # 避免推荐已经评估过的配置
            for config in challengers.challengers:
                if config not in self.history.configurations:
                    cur_config = config
                    recommend_flag = True
                    break

            if recommend_flag:
                logger.warn("Successfully recommend a configuration through Advisor!")
            else:
                logger.error("Failed to recommend am unique configuration ! Return a random config")
                cur_config = self.sample_random_configs(self.sample_space, 1, excluded_configs=self.history.configurations)

        logger.info("ret conf: %s" % (str(cur_config)))
        return cur_config

    
    '''
    初始化空间压缩，更新优化器
    indices 表示把原搜索空间压缩到只有这个列表中这些维度
    '''
    def compress_space_optimizer(self, space_history=None):        
        self.compressor, indices = None, None
        
        if self.cprs_strategy == 'expert':
            indices = sorted([self.origin_config_space.get_idx_by_hyperparameter_name(param) for param in self.expert_params])          
        elif self.cprs_strategy != 'none':
            if space_history is None:
                logger.warning('No compress history data provided, using original ConfigurationSpace.')
                indices = list(range(len(self.origin_config_space)))  # 直接使用完整空间
            else:
                hist_x = []
                hist_y = []
                for idx, (X, y) in enumerate(space_history):
                    if not idx:
                        logger.warn("Get objectives of space_history[0]: %s" % str(np.array(y)))
                    hist_x.append(convert_configurations_to_array(X))
                    hist_y.append(np.array(y))
                if hist_x and hist_y:
                    self.compressor, indices = build_my_compressor(
                        hist_x,
                        hist_y,
                        self.cp_topk,
                        func_str=self.cprs_strategy
                    )

                    new_indices = [self.origin_config_space.get_idx_by_hyperparameter_name(param) for param in self.expert_params]

                    keep_names = [self.origin_config_space.get_hyperparameter_by_idx(i) for i in new_indices]
                    logger.info(f"[Compression] Keep-by-expert parameters: {keep_names}")

                    for idx in indices:
                        if idx not in new_indices:
                            new_indices.append(idx)
                            if len(new_indices) == self.cp_topk:
                                break
                    indices = sorted(new_indices)
                else:
                    logger.warning("Invalid space history data, using original ConfigurationSpace.")
                    indices = list(range(len(self.origin_config_space)))

        self._set_space_by_indices(indices)
        self._set_range_compression()
        self.history.config_space = self.sample_space
        self.history.meta_info["config_space"] = js.loads(write(self.config_space))
        self.history.meta_info["sample_space"] = js.loads(write(self.sample_space))

        self.ini_configs = list()
        self.default_config = self.sample_space.get_default_configuration()
        self.default_config.origin = "default"
        logger.info("ConfigSpace after whole compression (dimension + range): %s !!!" % (str(self.sample_space)))
        
        self.param_names = [param.name for param in self.config_space.get_hyperparameters()]
        logger.info("Compressed configuration space: %s" % (self.param_names))

        
        self.surrogate = build_my_surrogate(func_str=self.surrogate_type, config_space=self.config_space, rng=self.rng,
                                            transfer_learning_history=self._source_hpo_data_in_new_space(self.source_hpo_data),
                                            extra_dim=self.extra_dim, norm_y=self.norm_y,)
        _kwargs = self.ep_args if self.ep_strategy in ['pibo', 'bo_pro'] else {}
        self.acq_func = build_my_acq_func(func_str=self.acq_type, model=self.surrogate, **_kwargs)
            
        self.acq_optimizer = InterleavedLocalAndRandomSearch(acquisition_function=self.acq_func,
                                                             rand_prob=self.rand_prob, rand_mode=self.rand_mode, rng=self.rng,
                                                             config_space=self.expert_modified_space if self.expert_modified_space is not None else self.sample_space)
        

    def _set_range_compression(self):
        if not hasattr(self, "sample_space") or self.sample_space is None:
            self.sample_space = copy.deepcopy(self.config_space)

        if self.range_config_space is not None:
            logger.info("Begin to make range compression...")
            range_hp_names = [hp.name for hp in self.range_config_space.get_hyperparameters()]

            for hp in list(self.sample_space.get_hyperparameters()):
                name = hp.name
                if name in range_hp_names:
                    new_hp = self.range_config_space.get_hyperparameter(name)
                    if hasattr(new_hp, 'lower') and hasattr(new_hp, 'upper'):
                        if isinstance(new_hp, sp.Int):
                            new_hp_obj = sp.Int(
                                name=name,
                                lower=new_hp.lower,
                                upper=new_hp.upper,
                                default_value=new_hp.default_value,
                                log=new_hp.log,
                            )
                        elif isinstance(new_hp, sp.Real):
                            new_hp_obj = sp.Real(
                                name=name,
                                lower=new_hp.lower,
                                upper=new_hp.upper,
                                default_value=new_hp.default_value,
                                log=new_hp.log,
                            )
                        else:
                            logger.warning(f"Unsupported hp type (numeric) for {name}")
                            continue
                        self.sample_space._hyperparameters.pop(name)
                        self.sample_space.add_hyperparameter(new_hp_obj)
                        logger.info(
                            f"Range compressed for [{name}]: "
                            f"[{new_hp.lower}, {new_hp.upper}], default={new_hp.default_value}"
                        )
                        # origin 只改 default，不动范围
                        ori_hp = self.origin_config_space.get_hyperparameter(name)
                        ori_hp.default_value = new_hp.default_value

                    elif hasattr(new_hp, 'choices'):
                        new_hp_obj = sp.Categorical(
                            name=name,
                            choices=list(new_hp.choices),
                            default_value=new_hp.default_value,
                        )
                        self.sample_space._hyperparameters.pop(name)
                        self.sample_space.add_hyperparameter(new_hp_obj)
                        logger.info(
                            f"Choices compressed for [{name}]: "
                            f"{new_hp.choices}, default={new_hp.default_value}"
                        )
                        ori_hp = self.origin_config_space.get_hyperparameter(name)
                        ori_hp.default_value = new_hp.default_value
                    else:
                        logger.warning(
                            f"Skip {name}, unsupported hyperparameter type for range compression."
                        )
        else:
            logger.info("No need to make range compression!!!")


    '''
    每次进行空间重压缩的时候调用，更新优化器
    '''
    def _set_space_by_indices(self, indices):
        if not indices:  # 如果 indices 为空或 None，直接使用原空间
            logger.info(f"Using original ConfigurationSpace (no compression).")
            self.config_space = copy.deepcopy(self.origin_config_space)
            self.fixed_indices = []
            self.current_indices = list(range(len(self.config_space)))
        else:
            logger.info(f"Use algorithm [{self.cprs_strategy}] to reset search space, only use indices: {indices}")
            self.current_indices = indices
            self.fixed_indices = [i for i in range(len(self.origin_config_space)) if i not in indices]
            self.config_space = ConfigurationSpace()
            for idx in indices:
                name = self.origin_config_space.get_hyperparameter_by_idx(idx)
                if name in self.origin_config_space:
                    self.config_space.add_hyperparameter(self.origin_config_space[name])
                    
            if self.origin_expert_space is not None:
                self.expert_modified_space = ConfigurationSpace()
                for idx in indices:
                    name = self.origin_expert_space.get_hyperparameter_by_idx(idx)
                    if name in self.origin_expert_space:
                        self.expert_modified_space.add_hyperparameter(self.origin_expert_space[name])

        logger.info("Currently fixed indices: %s" % (self.fixed_indices))
        

    def _source_hpo_data_in_new_space(self, source_hpo_data):
        if source_hpo_data is None:
            return None

        new_his = []
        for his in source_hpo_data:
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
                for name in self.param_names:
                    new_conf[name] = obs['config'][name]
                obs['config'] = new_conf
                
            global_start_time = data.pop('global_start_time')
            global_start_time = datetime.fromisoformat(global_start_time)
            observations = data.pop('observations')
            observations = [Observation.from_dict(obs, self.config_space) for obs in observations]

            history = History(**data)
            history.global_start_time = global_start_time
            history.update_observations(observations)
            
            new_his.append(history)
        
        return new_his        
    
    def compute_range_compression(self, old_data_path=None, new_data_path=None, 
                                 range_compress_data_path=None, json_file=None):
        """
        计算范围压缩空间
        
        Args:
            old_data_path: 历史数据路径
            range_compress_data_path: 范围压缩数据路径
            json_file: 配置文件路径
            
        Returns:
            ConfigurationSpace: 压缩后的配置空间
        """
        try:
            from space_values import haisi_huge_spaces_from_json
            
            if range_compress_data_path is None:
                from config import RANGE_COMPRESS_DATA
                range_compress_data_path = RANGE_COMPRESS_DATA
            if json_file is None:
                from config import HUGE_SPACE_FILE
                json_file = HUGE_SPACE_FILE
            
            logger.info("开始计算范围压缩空间...")
            old_space, new_space = haisi_huge_spaces_from_json(
                old_data_path=range_compress_data_path,
                json_file=json_file
            )
            
            logger.info(f"范围压缩完成：原始空间参数 {len(old_space.get_hyperparameters())} 个，"
                       f"压缩后空间参数 {len(new_space.get_hyperparameters())} 个")
            
            return new_space
            
        except ImportError as e:
            logger.warning(f"无法导入范围压缩模块: {e}")
            logger.info("将使用原始配置空间")
            return self.origin_config_space
        except Exception as e:
            logger.error(f"计算范围压缩时出错: {e}")
            logger.info("将使用原始配置空间")
            return self.origin_config_space