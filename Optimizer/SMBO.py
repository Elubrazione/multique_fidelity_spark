from .base import BaseOptimizer

class SMBO(BaseOptimizer):
    def __init__(self, config_space, eval_func, iter_num=200, per_run_time_limit=None,
                 method_id='smbo', task_id='test', target='redis',
                 ws_strategy='none', ws_args=None, tl_strategy='none', tl_args=None,
                 cprs_strategy='none', cp_args=None,
                 backup_flag=False, save_dir='./results',
                 seed=42, rand_prob=0.15, rand_mode='ran',
                 config_modifier = None, expert_modified_space=None, task_manager=None):

        super().__init__(config_space=config_space, eval_func=eval_func, iter_num=iter_num,
                         per_run_time_limit=per_run_time_limit,
                         method_id=method_id, task_id=task_id, target=target,
                         ws_strategy=ws_strategy, ws_args=ws_args, tl_strategy=tl_strategy, tl_args=tl_args,
                         cprs_strategy=cprs_strategy, cp_args=cp_args,
                         backup_flag=backup_flag, save_dir=save_dir,
                         seed=seed, rand_prob=rand_prob, rand_mode=rand_mode,
                         config_modifier = config_modifier, expert_modified_space=expert_modified_space, task_manager=task_manager)
