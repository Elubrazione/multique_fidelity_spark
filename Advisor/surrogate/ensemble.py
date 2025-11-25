import numpy as np
from openbox import logger
from ConfigSpace import ConfigurationSpace

from .base import TransferLearningSurrogate
from .weight import MFGPEWeightCalculator, RGPEWeightCalculator


class MFGPE(TransferLearningSurrogate):
    
    def __init__(
        self, 
        config_space: ConfigurationSpace, 
        source_hpo_data=None, 
        seed=0,
        surrogate_type='prf', 
        num_src_hpo_trial=50, 
        only_source=False, 
        norm_y=True, 
        **kwargs
    ):
        weight_calculator = MFGPEWeightCalculator(n_power=3)
        
        super().__init__(
            config_space=config_space,
            surrogate_type=surrogate_type,
            rng=np.random.RandomState(seed),
            only_source=only_source,
            num_src_trials=num_src_hpo_trial,
            source_data=source_hpo_data,
            weight_calculator=weight_calculator,
            norm_y=norm_y,
            **kwargs
        )
        self.method_id = 'mfgpe'


class RGPE(TransferLearningSurrogate):
    def __init__(
        self, 
        config_space: ConfigurationSpace, 
        source_hpo_data=None, 
        seed=0,
        surrogate_type='prf', 
        num_src_hpo_trial=50, 
        only_source=False, 
        norm_y=True, 
        num_sample: int = 50,
        **kwargs
    ):
        weight_calculator = RGPEWeightCalculator(
            num_sample=num_sample,
            use_dilution=True
        )
        
        super().__init__(
            config_space=config_space,
            source_data=source_hpo_data,
            surrogate_type=surrogate_type,
            only_source=only_source,
            num_src_trials=num_src_hpo_trial,
            weight_calculator=weight_calculator,
            rng=np.random.RandomState(seed),
            norm_y=norm_y,
            **kwargs
        )
        self.method_id = 'rgpe'
