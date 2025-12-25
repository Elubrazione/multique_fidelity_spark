import numpy as np
from typing import List, Dict, Tuple, Optional
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace, Configuration
from scipy.stats import spearmanr
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler


class IICP:
    def __init__(self, min_samples: int = 5, scc_threshold: float = 0.2, 
                 kpca_kernel: str = 'rbf', n_components: Optional[int] = None):
        self.min_samples = min_samples
        self.scc_threshold = scc_threshold
        self.kpca_kernel = kpca_kernel
        self.n_components = n_components
        
        self.config_space: Optional[ConfigurationSpace] = None
        self.param_names: List[str] = []
        self.samples: List[Dict] = []
        self.r_conf_param_names: List[str] = []
        self.kpca: Optional[KernelPCA] = None
        self.scaler: Optional[StandardScaler] = None
        self.compressed_param_indices: Optional[List[int]] = None
        
    def extract_samples_from_history(self,
                                    history: History,
                                    config_space: ConfigurationSpace,
                                    data_size: float = 1.0) -> List[Dict]:
        logger.info(f"IICP: Extracting samples from History ({len(history)} observations)...")
        
        samples = []
        self.config_space = config_space
        self.param_names = list(config_space.get_hyperparameter_names())
        
        for obs in history.observations:
            objectives = getattr(obs, 'objectives', None)
            if objectives and len(objectives) > 0:
                ti = objectives[0] if np.isfinite(objectives[0]) else float('inf')
            else:
                ti = float('inf')
            
            config = getattr(obs, 'config', None)
            if config is None:
                continue
            
            confi = config.get_dictionary()
            
            samples.append({
                'ti': ti,
                'confi': confi,
                'ds': data_size,
                'config': config
            })
        
        valid_samples = sum(1 for s in samples if np.isfinite(s['ti']))
        logger.info(f"IICP: Extracted {len(samples)} samples, {valid_samples} valid")
        
        self.samples = samples
        return samples
    
    def cps(self, samples: List[Dict]) -> List[str]:
        logger.info("IICP: Performing CPS (Configuration Parameter Selection)...")
        
        if not samples:
            logger.warning("IICP: No samples available for CPS")
            return self.param_names
        
        execution_times = []
        config_values = []
        
        for sample in samples:
            ti = sample['ti']
            if not np.isfinite(ti):
                continue
            
            execution_times.append(ti)
            confi = sample['confi']
            
            config_array = []
            for param_name in self.param_names:
                value = confi.get(param_name)
                if isinstance(value, (int, float)):
                    config_array.append(value)
                elif isinstance(value, str):
                    hp = self.config_space[param_name]
                    if hasattr(hp, 'choices'):
                        try:
                            config_array.append(hp.choices.index(value))
                        except ValueError:
                            config_array.append(0)
                    else:
                        config_array.append(0)
                else:
                    config_array.append(0)
            
            config_values.append(config_array)
        
        if len(execution_times) < 1:
            logger.warning("IICP: Insufficient valid samples for CPS, using all parameters")
            return self.param_names
        
        execution_times = np.array(execution_times)
        config_values = np.array(config_values)
        
        param_scc = {}
        for idx, param_name in enumerate(self.param_names):
            param_values = config_values[:, idx]
            
            if np.std(param_values) == 0:
                param_scc[param_name] = 0.0
                continue
            
            try:
                scc, p_value = spearmanr(param_values, execution_times)
                param_scc[param_name] = abs(scc) if not np.isnan(scc) else 0.0
            except Exception as e:
                logger.warning(f"IICP: Failed to compute SCC for {param_name}: {e}")
                param_scc[param_name] = 0.0
        
        r_conf_params = [
            param_name for param_name, scc in param_scc.items()
            if scc >= self.scc_threshold
        ]
        
        logger.info(f"IICP: CPS selected {len(r_conf_params)}/{len(self.param_names)} parameters")
        logger.info(f"IICP: Removed parameters: {[p for p in self.param_names if p not in r_conf_params][:10]}{'...' if len(self.param_names) - len(r_conf_params) > 10 else ''}")
        
        self.r_conf_param_names = r_conf_params
        return r_conf_params
    
    def cpe(self, samples: List[Dict], r_conf_params: List[str]) -> Tuple[np.ndarray, KernelPCA]:
        logger.info("IICP: Performing CPE (Configuration Parameter Extraction) with KPCA...")
        
        if not samples or not r_conf_params:
            logger.warning("IICP: No samples or parameters for CPE")
            return np.array([]), None
        
        config_values = []
        valid_indices = []
        
        for idx, sample in enumerate(samples):
            ti = sample['ti']
            if not np.isfinite(ti):
                continue
            
            confi = sample['confi']
            config_array = []
            
            for param_name in r_conf_params:
                value = confi.get(param_name)
                if isinstance(value, (int, float)):
                    config_array.append(value)
                elif isinstance(value, str):
                    hp = self.config_space[param_name]
                    if hasattr(hp, 'choices'):
                        try:
                            config_array.append(hp.choices.index(value))
                        except ValueError:
                            config_array.append(0)
                    else:
                        config_array.append(0)
                else:
                    config_array.append(0)
            
            config_values.append(config_array)
            valid_indices.append(idx)
        
        if len(config_values) < 1:
            logger.warning("IICP: Insufficient valid samples for CPE")
            return np.array([]), None
        
        config_values = np.array(config_values)
        
        self.scaler = StandardScaler()
        config_values_scaled = self.scaler.fit_transform(config_values)
        
        n_components = self.n_components
        if n_components is None:
            max_components = min(config_values_scaled.shape[0] - 1, config_values_scaled.shape[1])
            n_components = min(max_components, max(1, int(0.8 * config_values_scaled.shape[1])))
        
        try:
            self.kpca = KernelPCA(
                n_components=n_components,
                kernel=self.kpca_kernel,
                fit_inverse_transform=True  
            )
            transformed_features = self.kpca.fit_transform(config_values_scaled)
            
            logger.info(f"IICP: CPE extracted {n_components} components from {len(r_conf_params)} parameters")
            logger.info(f"IICP: KPCA kernel: {self.kpca_kernel}, explained variance ratio: {self.kpca.lambdas_[:5] if hasattr(self.kpca, 'lambdas_') else 'N/A'}")
            
            return transformed_features, self.kpca
            
        except Exception as e:
            logger.error(f"IICP: Failed to apply KPCA: {e}")
            return config_values_scaled, None
    
    def analyze(self,
                history: History,
                config_space: ConfigurationSpace,
                data_size: float = 1.0) -> Tuple[List[str], Optional[KernelPCA]]:
        logger.info("IICP: Starting important configuration parameter identification...")
        
        self.config_space = config_space
        self.param_names = list(config_space.get_hyperparameter_names())
        
        if len(history) < self.min_samples:
            logger.warning(f"IICP: Insufficient samples in History ({len(history)} < {self.min_samples}). "
                         f"Using all parameters.")
            self.r_conf_param_names = self.param_names
            self.kpca = None
            return self.param_names, None
        
        samples = self.extract_samples_from_history(
            history=history,
            config_space=config_space,
            data_size=data_size
        )
        
        r_conf_params = self.cps(samples)
        
        if not r_conf_params:
            logger.warning("IICP: No parameters selected after CPS")
            self.r_conf_param_names = []
            self.kpca = None
            return [], None
        
        transformed_features, kpca = self.cpe(samples, r_conf_params)
        self.kpca = kpca
        
        logger.info("IICP: Analysis complete")
        logger.info(f"IICP: Selected {len(r_conf_params)} parameters after CPS")
        if kpca is not None:
            logger.info(f"IICP: Extracted {transformed_features.shape[1]} components after CPE")
        
        return r_conf_params, kpca
    
    def transform_config(self, config: Configuration) -> np.ndarray:
        if self.kpca is None or self.scaler is None:
            raise ValueError("IICP: KPCA model not trained. Call analyze() first.")
        
        config_dict = config.get_dictionary()
        config_array = []
        
        for param_name in self.r_conf_param_names:
            value = config_dict.get(param_name)
            if isinstance(value, (int, float)):
                config_array.append(value)
            elif isinstance(value, str):
                hp = self.config_space[param_name]
                if hasattr(hp, 'choices'):
                    try:
                        config_array.append(hp.choices.index(value))
                    except ValueError:
                        config_array.append(0)
                else:
                    config_array.append(0)
            else:
                config_array.append(0)
        
        config_array = np.array(config_array).reshape(1, -1)
        
        config_scaled = self.scaler.transform(config_array)
        transformed = self.kpca.transform(config_scaled)
        
        return transformed.flatten()
    
    def inverse_transform(self, transformed_features: np.ndarray) -> np.ndarray:
        if self.kpca is None or self.scaler is None:
            raise ValueError("IICP: KPCA model not trained. Call analyze() first.")
        
        original_features = self.kpca.inverse_transform(transformed_features.reshape(1, -1))
        original_features = self.scaler.inverse_transform(original_features)
        
        return original_features.flatten()
    
    def get_analysis_info(self) -> Dict:
        return {
            'min_samples': self.min_samples,
            'scc_threshold': self.scc_threshold,
            'kpca_kernel': self.kpca_kernel,
            'original_params': len(self.param_names) if self.param_names else 0,
            'cps_selected_params': len(self.r_conf_param_names) if self.r_conf_param_names else 0,
            'r_conf_param_names': self.r_conf_param_names,
            'kpca_n_components': self.kpca.n_components if self.kpca is not None else None,
        }
