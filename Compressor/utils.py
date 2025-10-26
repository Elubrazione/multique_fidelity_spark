import json
import copy
import shap
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.ensemble import RandomForestRegressor
from ConfigSpace import ConfigurationSpace, Configuration
from openbox import space as sp, logger
from openbox.utils.config_space.util import convert_configurations_to_array


def create_param(key, value):
    q_val = value.get('q', None)
    param_type = value['type']

    if param_type == 'integer':
        return sp.Int(key, value['min'], value['max'], default_value=value['default'], q=q_val)
    elif param_type == 'real':
        return sp.Real(key, value['min'], value['max'], default_value=value['default'], q=q_val)
    elif param_type == 'enum': 
        return sp.Categorical(key, value['enum_values'], default_value=value['default'])
    elif param_type == 'categorical':
        return sp.Categorical(key, value['choices'], default_value=value['default'])
    else:
        raise ValueError(f"Unsupported type: {param_type}")

def parse_combined_space(json_file_origin, json_file_new):
    if isinstance(json_file_origin, str):
        with open(json_file_origin, 'r') as f:
            conf = json.load(f)
        space = sp.Space()
        for key, value in conf.items():
            if key not in space.keys():
                para = create_param(key, value)
                space.add_variable(para)
    else:
        space = copy.deepcopy(json_file_origin)

    if isinstance(json_file_new, str):
        with open(json_file_new, 'r') as f:
            conf_new = json.load(f)
        for key, value in conf_new.items():
            if key not in space.keys():
                para = create_param(key, value)
                space.add_variable(para)
    else:
        for param in json_file_new.get_hyperparameters():
            if param.name not in space.keys():
                space.add_variable(param)
    
    return space

def create_space_from_ranges(original_space: ConfigurationSpace, compressed_ranges: Dict[str, Tuple[float, float]]) -> ConfigurationSpace:
    """Create compressed configuration space from ranges."""
    compressed_space = copy.deepcopy(original_space)
    
    for param_name, (min_val, max_val) in compressed_ranges.items():
        try:
            hp = compressed_space.get_hyperparameter(param_name)
            if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                # Validate range before updating
                if min_val >= max_val:
                    logger.warning(f"Invalid range [{min_val}, {max_val}] for {param_name}, skipping compression")
                    continue
                    
                _update_numeric_hp(hp, min_val, max_val)
                logger.info(f"Compressed {param_name}: [{hp.lower}, {hp.upper}]")
        except Exception as e:
            logger.warning(f"Failed to compress parameter {param_name}: {e}")
            
    return compressed_space


def load_performance_data(data_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(data_path)
        logger.info(f"Loaded {len(data)} records from {data_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load data from {data_path}: {e}")
        return None

def prepare_historical_data(
    space_history: List[Tuple[List[Configuration], List[float]]]
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Prepare historical data for compression analysis.
    
    Args:
        space_history: List of Tuple[List[Configuration], List[float]] tuples
        
    Returns:
        Tuple of (hist_x, hist_y) where hist_x is list of configurations,
        hist_y is list of objective arrays
    """
    try:
        hist_x = []
        hist_y = []
        for idx, (X, y) in enumerate(space_history):
            if not idx:
                logger.info(f"Processing space_history[0] objectives: {np.array(y)}")
            hist_x.append(convert_configurations_to_array(X))
            hist_y.append(np.array(y))
        return hist_x, hist_y
    except Exception as e:
        logger.error(f"Error preparing historical data: {e}")
        return [], []

def load_expert_params(expert_config_file: str, key: str) -> List[str]:
    """Load expert parameters from configuration file."""
    try:
        with open(expert_config_file, "r") as f:
            all_expert_params = json.load(f)   
        expert_params = all_expert_params.get(key, [])             
        return expert_params
        
    except FileNotFoundError:
        logger.warning(f"Expert config file not found: {expert_config_file}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing expert config file: {e}")
        return []
    except Exception as e:
        logger.error(f"Error loading expert parameters: {e}")
        return []


def _update_numeric_hp(hp, min_val: float, max_val: float) -> None:
    """Update numeric hyperparameter bounds and default value."""
    if isinstance(hp, sp.Int):
        hp.lower = int(min_val)
        hp.upper = int(max_val)
        if not (hp.lower <= hp.default_value <= hp.upper):
            hp.default_value = (hp.lower + hp.upper) // 2
    elif isinstance(hp, sp.Real):
        hp.lower = float(min_val)
        hp.upper = float(max_val)
        if not (hp.lower <= hp.default_value <= hp.upper):
            hp.default_value = (hp.lower + hp.upper) / 2


def _filter_range_by_std(data: List[float], sigma: float = 2.0) -> Tuple[float, float]:
    """Get range based on standard deviation filtering."""
    data_array = np.array(data)
    mean = np.mean(data_array)
    std = np.std(data_array)
    
    min_val = max(np.min(data_array), mean - sigma * std)
    max_val = min(np.max(data_array), mean + sigma * std)
    
    if min_val >= max_val:
        min_val = np.min(data_array)
        max_val = np.max(data_array)
        if min_val >= max_val:
            min_val = mean - 0.1
            max_val = mean + 0.1
    return min_val, max_val


def update_hp_range(space: ConfigurationSpace, name: str, new_hp: Any) -> None:
    """Update the range of a specific hyperparameter in place."""
    if name not in space._hyperparameters:
        logger.warning(f"Hyperparameter '{name}' not found in space")
        return
    
    if hasattr(new_hp, 'lower') and hasattr(new_hp, 'upper'):   # Numeric hyperparameter
        # Validate range before creating new hyperparameter
        if new_hp.upper <= new_hp.lower:
            logger.warning(f"Invalid range [{new_hp.lower}, {new_hp.upper}] for {name}, skipping compression")
            return
            
        if isinstance(new_hp, sp.Int):
            new_hp_obj = sp.Int(
                name=name,
                lower=int(new_hp.lower),
                upper=int(new_hp.upper),
                default_value=int(new_hp.default_value),
                log=new_hp.log,
            )
        elif isinstance(new_hp, sp.Real):
            new_hp_obj = sp.Real(
                name=name,
                lower=float(new_hp.lower),
                upper=float(new_hp.upper),
                default_value=float(new_hp.default_value),
                log=new_hp.log,
            )
        else:
            logger.warning(f"Unsupported numeric hyperparameter type for {name}")
            return
            
        space._hyperparameters.pop(name)
        space.add_hyperparameter(new_hp_obj)
        logger.info(
            f"Range compressed [{name}]: [{new_hp.lower}, {new_hp.upper}], "
            f"default={new_hp.default_value}"
        )
            
    elif hasattr(new_hp, 'choices'):    # Categorical hyperparameter
        new_hp_obj = sp.Categorical(
            name=name,
            choices=new_hp.choices,
            default_value=new_hp.default_value,
        )
        space._hyperparameters.pop(name)
        space.add_hyperparameter(new_hp_obj)
        logger.info(
            f"Range compressed [{name}]: {list(new_hp.choices)}, "
            f"default={new_hp.default_value}"
        )
    else:
        logger.warning(f"Unsupported hyperparameter type for {name}")


def collect_compression_details(original_space: ConfigurationSpace, compressed_space: ConfigurationSpace) -> Dict[str, Any]:
    """Collect detailed information about range compression."""
    details = {}
    range_hp_names = [hp.name for hp in compressed_space.get_hyperparameters()]
    
    for hp in original_space.get_hyperparameters():
        name = hp.name
        if name in range_hp_names:
            original_hp = hp
            compressed_hp = compressed_space.get_hyperparameter(name)
            
            if hasattr(original_hp, 'lower') and hasattr(original_hp, 'upper'): # Numeric hyperparameter
                details[name] = {
                    'type': 'numeric',
                    'original_range': [original_hp.lower, original_hp.upper],
                    'compressed_range': [compressed_hp.lower, compressed_hp.upper],
                    'original_default': original_hp.default_value,
                    'compressed_default': compressed_hp.default_value,
                    'compression_ratio': (compressed_hp.upper - compressed_hp.lower) / (original_hp.upper - original_hp.lower)
                }
            elif hasattr(original_hp, 'choices'):   # Categorical hyperparameter
                details[name] = {
                    'type': 'categorical',
                    'original_choices': list(original_hp.choices),
                    'compressed_choices': list(compressed_hp.choices),
                    'original_default': original_hp.default_value,
                    'compressed_default': compressed_hp.default_value,
                    'compression_ratio': len(compressed_hp.choices) / len(original_hp.choices)
                }
    return details


def compute_shap_based_ranges(X, feature_cols, shap_vals_array, sigma, original_space=None):
    """Compute ranges based on SHAP analysis."""
    shap_based_ranges = {}
    for i, param in enumerate(feature_cols):
        # X is a numpy array, not DataFrame, so we can use [:, i] to get the i-th column
        values = X[:, i]
        shap_effect = shap_vals_array[:, i]
        
        # Get values with beneficial SHAP effects (negative for minimization)
        beneficial_values = values[shap_effect < 0]
        
        if len(beneficial_values) > 0:
            min_val, max_val = _filter_range_by_std(beneficial_values, sigma)
        else:
            min_val, max_val = _filter_range_by_std(values, sigma)
        
        # If we have original space, convert normalized values back to original parameter ranges
        if original_space is not None:
            try:
                hp = original_space.get_hyperparameter(param)
                if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                    # Convert normalized values back to original parameter ranges
                    original_min = hp.lower + min_val * (hp.upper - hp.lower)
                    original_max = hp.lower + max_val * (hp.upper - hp.lower)
                    min_val, max_val = original_min, original_max
            except Exception as e:
                logger.warning(f"Failed to convert normalized range for {param}: {e}")
        
        shap_based_ranges[param] = (min_val, max_val)
    return shap_based_ranges


def filter_numeric_params(space: ConfigurationSpace) -> List[str]:
    """Get list of numeric parameter names from configuration space."""
    numeric_params = []
    for hp in space.get_hyperparameters():
        if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
            numeric_params.append(hp.name)
    return numeric_params