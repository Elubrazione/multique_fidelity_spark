from openbox import logger

from config import ConfigManager
from Evaluator.executor import ExecutorManager
from Optimizer import get_optimizer
from Optimizer.utils import load_space_from_json
from task_manager import TaskManager
from Compressor import get_compressor

args = ConfigManager.parse_args()
config_manager = ConfigManager(config_file=args.config, args=args)
config_space = load_space_from_json(config_manager.config_space)

logger_kwargs = config_manager.get_logger_kwargs(args.task, args.opt, args.log_level)
logger.init(**logger_kwargs)
logger_kwargs.update({'force_init': False})

executor = ExecutorManager(
    config_space=config_space,
    test_mode=args.test_mode,
    debug=args.debug,
    config_manager=config_manager
)

# Create task_manager with config_manager
task_manager = TaskManager.instance(
    config_space=config_space,
    config_manager=config_manager,
    logger_kwargs=logger_kwargs
)
task_manager.calculate_meta_feature(
    eval_func=executor, task_id=args.task,
    test_mode=args.test_mode, resume=args.resume
)

cp_args = config_manager.get_cp_args(config_space)
compressor = get_compressor(
    compressor_type=cp_args.get('strategy', 'none'),
    config_space=config_space,
    **cp_args
)
task_manager.register_compressor(compressor)

opt_kwargs = {
    'config_space': config_space,
    'eval_func': executor,
    'config_manager': config_manager
}
optimizer = get_optimizer(args, **opt_kwargs)

if __name__ == '__main__':
    for i in range(optimizer.iter_num):
        optimizer.run_one_iter()