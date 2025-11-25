from manager import ConfigManager
from .base import BaseOptimizer

def get_optimizer(args, **kwargs):
    config_manager: ConfigManager = kwargs.get('config_manager')

    optimizer = BaseOptimizer(
        config_space=kwargs['config_space'],
        eval_func=kwargs['eval_func'],
        iter_num=args.iter_num,
        method_id=args.opt,
        task_id=args.task,
        backup_flag=args.backup_flag,
        ws_strategy=args.warm_start,
        tl_strategy=args.transfer,
        resume=args.resume,
        target=config_manager.target,
        save_dir=config_manager.save_dir,
        per_run_time_limit=kwargs.get('per_run_time_limit', None),
    )
    return optimizer


__all__ = ['get_optimizer']