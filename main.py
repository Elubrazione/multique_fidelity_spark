from openbox import logger

from manager import ConfigManager, TaskManager
from core.plugin_loader import PluginLoader
from Evaluator import MockExecutor
from Optimizer import get_optimizer
from Compressor.dimensio import get_compressor

args = ConfigManager.parse_args()
config_manager = ConfigManager(config_file=args.config, args=args)

target_system_name = config_manager.target_system
target_system = PluginLoader.load_target_system(target_system_name)
target_system.initialize(config_manager)

config_space = target_system.get_default_config_space()

logger_kwargs = config_manager.get_logger_kwargs(args.task, args.opt, args.log_level)
logger.init(**logger_kwargs)
logger_kwargs.update({'force_init': False})

evaluators = None
if args.test_mode:
    evaluators = [MockExecutor(seed=42)]

executor = target_system.get_evaluator_manager(
    config_space=config_space, 
    evaluators=evaluators,
    test_mode=args.test_mode
)


task_manager = TaskManager.instance(
    config_space=config_space,
    config_manager=config_manager,
    logger_kwargs=logger_kwargs,
    target_system=target_system
)
executor.attach_task_manager(task_manager)
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