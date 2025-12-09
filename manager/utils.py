import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from openbox import logger
from openbox.utils.history import History, Observation
from openbox.utils.constants import SUCCESS
from ConfigSpace import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    CategoricalHyperparameter
)


def create_hyperparameter_from_dict(hp_dict: Dict[str, Any]) -> Optional:
    hp_type = hp_dict.get('type', '').lower()
    name = hp_dict['name']
    
    if 'uniform_int' in hp_type or hp_type == 'int' or hp_type == 'integer':
        return UniformIntegerHyperparameter(
            name=name,
            lower=int(hp_dict['lower']),
            upper=int(hp_dict['upper']),
            default_value=hp_dict.get('default', int((hp_dict['lower'] + hp_dict['upper']) / 2)),
            log=hp_dict.get('log', False)
        )
    elif 'uniform_float' in hp_type or hp_type == 'float' or hp_type == 'real':
        return UniformFloatHyperparameter(
            name=name,
            lower=float(hp_dict['lower']),
            upper=float(hp_dict['upper']),
            default_value=hp_dict.get('default', (hp_dict['lower'] + hp_dict['upper']) / 2),
            log=hp_dict.get('log', False)
        )
    elif hp_type == 'categorical':
        return CategoricalHyperparameter(
            name=name,
            choices=hp_dict['choices'],
            default_value=hp_dict.get('default', hp_dict['choices'][0] if hp_dict['choices'] else None)
        )
    else:
        logger.warning(f"Unknown hyperparameter type: {hp_type} for {name}, skipping")
        return None


def create_config_space_for_params(
    param_names: Set[str],
    hyperparameters_def: List[Dict[str, Any]]
) -> ConfigurationSpace:
    cs = ConfigurationSpace()
    for hp_dict in hyperparameters_def:
        if hp_dict.get('name') in param_names:
            hp = create_hyperparameter_from_dict(hp_dict)
            if hp is not None:
                cs.add_hyperparameter(hp)
    return cs


def load_history_with_dynamic_space(
    filename: str,
    fallback_config_space: Optional[ConfigurationSpace] = None
) -> History:
    """
    Load history from JSON file, dynamically creating config space for each observation.
    
    This function handles cases where different observations may use different
    config spaces (e.g., first observation uses original space, later ones use
    compressed space with fewer parameters).
    
    Args:
        filename: Path to history JSON file
        fallback_config_space: Optional fallback config space if extraction fails
    
    Returns:
        History object with loaded observations
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    task_id = data.get('task_id', 'unknown_task')
    num_objectives = data.get('num_objectives', 1)
    num_constraints = data.get('num_constraints', 0)
    meta_info = data.get('meta_info', {})
    
    global_start_time_str = data.get('global_start_time')
    if global_start_time_str:
        global_start_time = datetime.fromisoformat(global_start_time_str)
    else:
        global_start_time = datetime.now()
    
    space_info = meta_info.get('space', {})
    original_space = space_info.get('original', {})
    hyperparameters_def = original_space.get('hyperparameters', [])
    
    if not hyperparameters_def:
        logger.warning(f"No hyperparameters definition found in {filename}, "
                      f"using fallback config space")
        if fallback_config_space is None:
            raise ValueError(f"Cannot load history: no hyperparameters definition "
                           f"and no fallback config space provided")
    else:
        logger.info(f"Found {len(hyperparameters_def)} hyperparameters in history file")
    
    observations_data = data.get('observations', [])
    observations = []
    
    config_space_cache: Dict[frozenset, ConfigurationSpace] = {}
    
    for obs_idx, obs_data in enumerate(observations_data):
        config_dict = obs_data.get('config', {})
        if not isinstance(config_dict, dict):
            raise ValueError(f"Observation {obs_idx} config must be a dictionary")
        
        param_names = set(config_dict.keys())
        param_key = frozenset(param_names)
        if param_key not in config_space_cache:
            if hyperparameters_def:
                config_space_cache[param_key] = create_config_space_for_params(
                    param_names, hyperparameters_def
                )
                logger.debug(f"Created config space for observation {obs_idx} "
                           f"with {len(param_names)} parameters: {sorted(param_names)}")
            else:
                config_space_cache[param_key] = fallback_config_space
        
        config_space = config_space_cache[param_key]
        
        try:
            config = Configuration(config_space, values=config_dict)
        except Exception as e:
            logger.error(f"Failed to create Configuration for observation {obs_idx}: {e}")
            logger.error(f"Config dict keys: {list(config_dict.keys())}")
            logger.error(f"Config space params: {[hp.name for hp in config_space.get_hyperparameters()]}")
            raise
        
        if 'objectives' in obs_data:
            objectives = obs_data['objectives']
        elif 'objective' in obs_data:
            objectives = [obs_data['objective']]
        else:
            raise ValueError(f"Observation {obs_idx} must have 'objectives' or 'objective' field")
        
        constraints = obs_data.get('constraints', None)
        
        trial_state = obs_data.get('trial_state', SUCCESS)
        if isinstance(trial_state, int):
            if trial_state == 0:
                trial_state = SUCCESS
        
        elapsed_time = obs_data.get('elapsed_time', 0.0)
        
        create_time_str = obs_data.get('create_time')
        if create_time_str:
            try:
                create_time = datetime.fromisoformat(create_time_str)
            except:
                create_time = None
        else:
            create_time = None
        
        extra_info = obs_data.get('extra_info', {})
        
        obs = Observation(
            config=config,
            objectives=objectives,
            constraints=constraints,
            trial_state=trial_state,
            elapsed_time=elapsed_time,
            extra_info=extra_info
        )
        
        if create_time:
            obs.create_time = create_time
        
        observations.append(obs)
    
    if observations:
        history_config_space = observations[0].config.configuration_space
    elif fallback_config_space:
        history_config_space = fallback_config_space
    else:
        history_config_space = ConfigurationSpace()
    
    history = History(
        task_id=task_id,
        num_objectives=num_objectives,
        num_constraints=num_constraints,
        config_space=history_config_space,
        meta_info=meta_info
    )
    
    history.global_start_time = global_start_time
    history.update_observations(observations)
    
    logger.info(f'Loaded history (len={len(observations)}) from {filename} '
                f'with dynamic config space support')
    
    return history

