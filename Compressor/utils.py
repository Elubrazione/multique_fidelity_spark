import json
import copy
import numpy as np
from openbox import space as sp, logger


def load_expert_params(scene: str, config_file: str = "configs/config_space/expert_space.json"):
    with open(config_file, "r") as f:
        all_params = json.load(f)
    if scene not in all_params:
        raise ValueError(f"Unknown scene: {scene}")
    
    # 对于spark场景，添加OS参数
    if scene in ["spark"] and "os" in all_params:
        expert_params = all_params[scene] + all_params["os"]
        logger.info(f"Added OS parameters to {scene} scene: {all_params['os']}")
        return expert_params
    
    return all_params[scene]

def create_param(key, value):
    q_val = value.get('q', None)
    param_type = value['type']

    if param_type == 'integer':
        return sp.Int(key, value['min'], value['max'],
                      default_value=value['default'], q=q_val)
    elif param_type == 'real':
        return sp.Real(key, value['min'], value['max'],
                       default_value=value['default'], q=q_val)
    elif param_type == 'enum': 
        return sp.Categorical(key, value['enum_values'],
                              default_value=value['default'])
    elif param_type == 'categorical':
        return sp.Categorical(key, value['choices'],
                              default_value=value['default'])
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

def build_my_compressor(X, y, k=23, func_str='none'):
    from shap import Explainer
    def _shap_importance(X, model):
        explainer = Explainer(model)
        # shap_values = np.abs(explainer(X).values).mean(axis=0)
        shap_values = np.abs(explainer(X,check_additivity=False).values).mean(axis=0)
        return shap_values

    importances = []
    models = []
    for i in range(len(X)):
        tX = X[i]
        ty = y[i]
        if func_str in ['ottertune', 'rover-l', 'rover', 'rover-s']:
            from xgboost import XGBRegressor
            model = XGBRegressor().fit(tX, ty)
            models.append(model)
            importances.append(_shap_importance(tX, model))
        elif func_str == 'rover-g':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor().fit(tX, ty)
            models.append(model)
            importances.append(model.feature_importances_)
        elif func_str == 'opadviser':
            from lightgbm import LGBMRegressor
            model = LGBMRegressor().fit(tX, ty)   # LGBM 不需要特征选择
            return model, list(range(tX.shape[1]))
        else:
            raise ValueError(f"Unknown func_str: {func_str}!")
    importances = np.mean(np.array(importances), axis = 0)
    k = min(k, len(importances))
    return models, sorted(np.argsort(-importances).tolist()[: k])