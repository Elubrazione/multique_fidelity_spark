import os
import argparse
import yaml
from typing import Dict, Any, Optional, List


class ConfigManager:
    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default='configs/base.yaml', help='Path to config YAML file')
        parser.add_argument('--opt', type=str, default='MFES_SMAC',
                            choices=['BOHB_GP', 'BOHB_SMAC', 'MFES_GP', 'MFES_SMAC', 'SMAC', 'GP', 'BOHB_SMAC'])
        parser.add_argument('--log_level', type=str, default='info', choices=['info', 'debug'])
        parser.add_argument('--iter_num', type=int, default=40)
        parser.add_argument('--R', type=int, default=27)
        parser.add_argument('--eta', type=int, default=3)
        
        parser.add_argument('--history_dir', type=str, default=None)
        parser.add_argument('--save_dir', type=str, default=None)
        parser.add_argument('--target', type=str, default=None)
        
        parser.add_argument('--compress', type=str, default='none', choices=['none', 'shap', 'expert'])
        parser.add_argument('--cp_topk', type=int, default=40)
        parser.add_argument('--warm_start', type=str, default='none', choices=['none', 'best_rover', 'best_all'])
        parser.add_argument('--ws_init_num', type=int, default=4)
        parser.add_argument('--ws_topk', type=int, default=4)
        parser.add_argument('--ws_inner_surrogate_model', type=str, default='prf')
        parser.add_argument('--transfer', type=str, default='none')
        parser.add_argument('--tl_topk', type=int, default=3)

        parser.add_argument('--backup_flag', action='store_true', default=False)        
        parser.add_argument('--task', type=str, default='test_ws')
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--rand_prob', type=float, default=0.15)
        parser.add_argument('--rand_mode', type=str, default='ran', choices=['ran', 'rs'])
        
        # debug mode: debug in server which has spark cluster, for quick testing
        # test_mode: code development mode when coding in local machine which has no spark cluster
        parser.add_argument('--test_mode', action='store_true', default=False)
        parser.add_argument('--debug', action='store_true', default=False)
        parser.add_argument('--resume', type=str, default=None)
        
        return parser.parse_args()

    def __init__(self, config_file='configs/base.yaml', args=None):
        self.config_file = config_file
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.config = self._load_config()
        self._apply_args_overrides(args)
    
    
    def _load_config(self) -> Dict[str, Any]:
        config_path = os.path.join(self.root_dir, self.config_file)
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if 'includes' in config:
            includes = config.pop('includes')
            merged_config = {}
            
            for include_file in includes:
                include_path = os.path.join(self.root_dir, include_file)
                if os.path.exists(include_path):
                    with open(include_path, 'r', encoding='utf-8') as f:
                        included_config = yaml.safe_load(f)
                        merged_config = self._merge_dict(merged_config, included_config)
            merged_config = self._merge_dict(merged_config, config)
            return merged_config
        
        return config
    
    def _merge_dict(self, base: Dict, override: Dict) -> Dict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # recursively merge dictionaries
                result[key] = self._merge_dict(result[key], value)
            else:
                result[key] = value
        return result
    
    def _apply_args_overrides(self, args) -> None:
        if args is None:
            return
        SKIP_ARGS = {'config', 'opt', 'task', 'log_level', 'iter_num', 'warm_start', 
                    'transfer', 'backup_flag', 'test_mode', 'debug', 'resume'}
        for arg_name, arg_value in vars(args).items():
            if arg_name in SKIP_ARGS:
                continue
            
            if self._should_override(arg_value):
                config_path = self._find_config_path(arg_name)
                if config_path:
                    self._set_nested_config(config_path, arg_value)
        
    
    def _find_config_path(self, key: str) -> Optional[List[str]]:
        PARAM_MAPPINGS = {
            'ws_init_num': ['method_args', 'ws_args', 'init_num'],
            'ws_topk': ['method_args', 'ws_args', 'topk'],
            'ws_inner_surrogate_model': ['method_args', 'ws_args', 'inner_surrogate_model'],
            'tl_topk': ['method_args', 'tl_args', 'topk'],
            'cp_topk': ['method_args', 'cp_args', 'topk'],
            'compress': ['method_args', 'cp_args', 'strategy'],
        }
        
        if key in PARAM_MAPPINGS:
            path = PARAM_MAPPINGS[key]
            current = self.config
            for k in path:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return None
            return path
        
        def search_recursive(current: Dict, path: List[str]) -> Optional[List[str]]:
            if key in current:
                return path + [key]
            for k, v in current.items():
                if isinstance(v, dict):
                    result = search_recursive(v, path + [k])
                    if result:
                        return result
            return None
        return search_recursive(self.config, [])
    
    def _should_override(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return True
        return value != '' and value != []
    
    
    def _set_nested_config(self, config_path: List[str], value: Any) -> None:
        current = self.config
        for key in config_path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[config_path[-1]] = value
    
    
    @property
    def paths(self) -> Dict[str, Any]:
        return self.config['paths']
    
    @property
    def local_cluster(self) -> Dict[str, Any]:
        return self.config['local_cluster']
    
    @property
    def multi_clusters(self) -> Dict[str, Any]:
        return self.config['multi_clusters']
    
    @property
    def method_args(self) -> Dict[str, Any]:
        return self.config['method_args']
    
    @property
    def config_spaces(self) -> Dict[str, Any]:
        return self.config['config_spaces']
    
    @property
    def log_dir(self) -> str:
        return os.path.join(self.root_dir, self.paths['log_dir'])
    
    @property
    def data_dir(self) -> str:
        return os.path.join(self.root_dir, self.paths['data_dir'])
    
    @property
    def history_dir(self) -> str:
        return os.path.join(self.root_dir, self.paths['history_dir'])

    @property
    def save_dir(self) -> str:
        return os.path.join(self.root_dir, self.paths['save_dir'])
    
    @property
    def database(self) -> str:
        return self.config['database']
    
    @property
    def target(self) -> str:
        return self.paths['target']
    
    @property
    def local_nodes(self) -> List[str]:
        return self.local_cluster['nodes']
    
    @property
    def local_server(self) -> str:
        return self.local_cluster['server']
    
    @property
    def local_username(self) -> str:
        return self.local_cluster['username']
    
    @property
    def local_password(self) -> str:
        return self.local_cluster['password']
    
    @property
    def multi_usernames(self) -> List[str]:
        return self.multi_clusters['usernames']
    
    @property
    def multi_passwords(self) -> List[str]:
        return self.multi_clusters['passwords']
    
    @property
    def multi_servers(self) -> List[str]:
        return self.multi_clusters['servers']
    
    @property
    def multi_nodes(self) -> List[List[str]]:
        return self.multi_clusters['nodes']

    @property
    def spark_log_dir(self) -> str:
        return self.local_cluster['spark_log_dir']
    
    @property
    def config_space(self) -> str:
        return os.path.join(
            self.root_dir, 
            self.config_spaces['config_space']
        )
    
    @property
    def expert_space(self) -> str:
        return os.path.join(
            self.root_dir,
            self.config_spaces['expert_space']
        )
    
    @property
    def similarity_threshold(self) -> float:
        return self.config['similarity_threshold']

    def get(self, key: str) -> Any:
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                if k not in value:
                    raise KeyError(f"Config key '{key}' not found (missing '{k}' in path)")
                value = value[k]
            else:
                raise KeyError(f"Config key '{key}' not found (expected dict at '{'.'.join(keys[:keys.index(k)])}', got {type(value).__name__})")
        return value

    
    def get_logger_kwargs(self, task: str, opt: str, log_level: str) -> Dict[str, Any]:
        logger_kwargs = self.method_args.get('logger_kwargs', {}).copy()
        logger_kwargs['name'] = task
        logger_kwargs['logdir'] = f'{self.log_dir}/{self.target}/{opt}'
        logger_kwargs['level'] = log_level.upper()
        return logger_kwargs
    
    def get_cp_args(self, config_space) -> Dict[str, Any]:
        from Compressor.utils import load_expert_params
        
        cp_args = self.method_args.get('cp_args', {}).copy()
        expert_params = load_expert_params(self.expert_space)
        cp_args['expert_params'] = [p for p in expert_params 
                                    if p in config_space.get_hyperparameter_names()]
        return cp_args