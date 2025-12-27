import numpy as np
from typing import List, Optional, Tuple
from openbox import logger
from ConfigSpace import ConfigurationSpace, Configuration
from openbox.utils.util_funcs import get_types
from openbox.utils.transform import zero_mean_unit_var_normalization, zero_mean_unit_var_unnormalization


class DAGP:
    def __init__(self, config_space: ConfigurationSpace, 
                 iicp_model=None,
                 kernel='rbf',
                 seed: int = 42):
        self.config_space = config_space
        self.iicp_model = iicp_model
        self.kernel = kernel
        self.seed = seed
        self.gp_model = None
        self.types = None
        self.bounds = None
        self.extra_dim = 1
        self.y_normalize_mean = None
        self.y_normalize_std = None
        self.norm_y = True
        
    def _init_types_bounds(self):
        if self.iicp_model is not None and self.iicp_model.kpca is not None:
            n_kpca_components = self.iicp_model.kpca.n_components
            self.types = np.zeros(n_kpca_components + self.extra_dim, dtype=np.uint)
            self.bounds = np.vstack([
                np.array([[-1.0, 1.0]] * n_kpca_components),
                np.array([[0.0, 1.0]] * self.extra_dim)
            ])
            logger.info(f"DAGP: Using IICP transformation, feature dim={n_kpca_components}+{self.extra_dim}={len(self.types)}")
        else:
            self.types, self.bounds = get_types(self.config_space)
            self.types = np.hstack((self.types, np.zeros(self.extra_dim, dtype=np.uint)))
            self.bounds = np.vstack((self.bounds, np.array([[0.0, 1.0]] * self.extra_dim)))
            logger.info(f"DAGP: Not using IICP, feature dim={len(self.types)}")
    
    def _build_gp_model(self):
        from openbox.surrogate.base.build_gp import create_gp_model
        
        if self.types is None or self.bounds is None:
            self._init_types_bounds()
        
        rng = np.random.RandomState(self.seed)
        model_type = self.kernel[:2] if len(self.kernel) >= 2 else 'gp'
        
        self.gp_model = create_gp_model(
            model_type=model_type,
            config_space=self.config_space,
            types=self.types,
            bounds=self.bounds,
            rng=rng
        )
        
        logger.info(f"DAGP: Built GP model with kernel {self.kernel}, "
                   f"input dim={len(self.types)} (config_features={len(self.types)-self.extra_dim}, data_size={self.extra_dim})")
    
    def _prepare_input(self, X_config: np.ndarray, data_sizes: np.ndarray) -> np.ndarray:
        if data_sizes.ndim == 1:
            data_sizes = data_sizes.reshape(-1, 1)
        return np.hstack([X_config, data_sizes])
    
    def _extract_config_features(self, configs: List[Configuration], 
                                 data_sizes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if data_sizes is None:
            data_sizes = np.ones(len(configs))
        
        if self.iicp_model is not None:
            X_config = np.array([self.iicp_model.transform_config(config) for config in configs])
        else:
            config_arrays = []
            for config in configs:
                config_dict = config.get_dictionary()
                config_array = []
                for hp_name in self.config_space.get_hyperparameter_names():
                    value = config_dict.get(hp_name)
                    hp = self.config_space[hp_name]
                    if isinstance(value, (int, float)):
                        config_array.append(value)
                    elif isinstance(value, str) and hasattr(hp, 'choices'):
                        try:
                            config_array.append(hp.choices.index(value))
                        except (ValueError, AttributeError):
                            config_array.append(0)
                    else:
                        config_array.append(0)
                config_arrays.append(config_array)
            X_config = np.array(config_arrays)
        
        return X_config, data_sizes
    
    def train(self, X: np.ndarray, y: np.ndarray, 
            data_sizes: Optional[np.ndarray] = None):
        if self.gp_model is None or self.types is None or self.bounds is None:
            if self.types is None or self.bounds is None:
                self._init_types_bounds()
            if self.gp_model is None:
                self._build_gp_model()
        
        if isinstance(X, list) and all(isinstance(x, Configuration) for x in X):
            X_config, data_sizes = self._extract_config_features(X, data_sizes)
            X_combined = self._prepare_input(X_config, data_sizes)
        elif isinstance(X, np.ndarray):
            if data_sizes is None:
                data_sizes = np.ones(X.shape[0])
            X_combined = self._prepare_input(X, data_sizes)
        else:
            raise ValueError(f"DAGP: Unsupported X type: {type(X)}")
        
        logger.info(f"DAGP: data_sizes shape={data_sizes.shape}, unique values={np.unique(data_sizes)}, "
                   f"min={np.min(data_sizes):.4f}, max={np.max(data_sizes):.4f}, mean={np.mean(data_sizes):.4f}")
        
        if self.norm_y:
            y, self.y_normalize_mean, self.y_normalize_std = zero_mean_unit_var_normalization(y)
        
        self.gp_model.train(X_combined, y)
        logger.info(f"DAGP: Trained on {len(y)} samples, input dim={X_combined.shape[1]}")
    
    def predict(self, X: np.ndarray, data_sizes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if self.gp_model is None:
            raise ValueError("DAGP: Model not trained. Call train() first.")
        
        if isinstance(X, list) and all(isinstance(x, Configuration) for x in X):
            X_config, data_sizes = self._extract_config_features(X, data_sizes)
            X_combined = self._prepare_input(X_config, data_sizes)
        elif isinstance(X, np.ndarray):
            if data_sizes is None:
                data_sizes = np.ones(X.shape[0])
            X_combined = self._prepare_input(X, data_sizes)
        
        mean, var = self.gp_model.predict(X_combined)
        
        if self.norm_y and self.y_normalize_mean is not None and self.y_normalize_std is not None:
            mean = zero_mean_unit_var_unnormalization(mean, self.y_normalize_mean, self.y_normalize_std)
            var = var * self.y_normalize_std ** 2
        
        return mean, var
    
    def predict_marginalized_over_instances(self, X: np.ndarray, 
                                            data_sizes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(X, np.ndarray) and len(X.shape) == 2:
            expected_feature_dim = len(self.types) - self.extra_dim if self.types is not None else None
            config_space_dim = len(self.config_space.get_hyperparameter_names())
            
            if (X.shape[1] == config_space_dim and 
                expected_feature_dim is not None and 
                X.shape[1] != expected_feature_dim):
                from ConfigSpace import Configuration
                
                hp_names = self.config_space.get_hyperparameter_names()
                configs = []
                for i in range(X.shape[0]):
                    config_dict = {}
                    for j, hp_name in enumerate(hp_names):
                        normalized_value = float(X[i, j])
                        hp = self.config_space.get_hyperparameter(hp_name)
                        
                        if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                            lower = hp.lower
                            upper = hp.upper
                            original_value = lower + normalized_value * (upper - lower)
                            
                            from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, NormalIntegerHyperparameter
                            if isinstance(hp, (UniformIntegerHyperparameter, NormalIntegerHyperparameter)):
                                original_value = int(round(original_value))
                                original_value = max(int(lower), min(int(upper), original_value))
                            else:
                                original_value = max(lower, min(upper, original_value))
                        elif hasattr(hp, 'choices'):
                            choices = list(hp.choices)
                            n_choices = len(choices)
                            idx = int(round(normalized_value * (n_choices - 1)))
                            idx = max(0, min(n_choices - 1, idx))
                            original_value = choices[idx]
                        else:
                            original_value = hp.default_value
                        
                        config_dict[hp_name] = original_value
                    
                    try:
                        config = Configuration(self.config_space, values=config_dict)
                        configs.append(config)
                    except Exception as e:
                        logger.warning(f"DAGP: Failed to create config from array: {e}")
                        if len(configs) > 0:
                            configs.append(configs[0])
                        else:
                            configs.append(self.config_space.get_default_configuration())
                
                mean, var = self.predict(configs, data_sizes)
            else:
                mean, var = self.predict(X, data_sizes)
        else:
            mean, var = self.predict(X, data_sizes)
        
        var_threshold = 1e-10
        var[var < var_threshold] = var_threshold
        var[np.isnan(var)] = var_threshold
        
        return mean, var
