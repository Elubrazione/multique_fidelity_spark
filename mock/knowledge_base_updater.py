from mock.history import fake_get_task_embedding, fake_get_task_best_config, get_history_data
from modules.knowledge_base_updater import update_knob_detail
from modules.regression_model import PerformanceModel
from config.knobs_list import *
import numpy as np
from util import check_sample, add_scale_dict

def fake_update_history(update_task_id, epoch_id, logger, weights=None):
    logger.info(f"Config {epoch_id} for history task {update_task_id} generation starts.")
    task_embedding = fake_get_task_embedding(update_task_id)
    if task_embedding is None:
        print(f"No history data is found for task {update_task_id}...")
        return None
    
    best_config = fake_get_task_best_config(update_task_id)
    if best_config is None:
        print(f"No best configuration is found for task {update_task_id}...")
        return None
    task_best_config = best_config
    updated_knob_details, core_thresholds, memory_thresholds = update_knob_detail(task_best_config)
    logger.info(f"Updated resource thresholds for task {update_task_id}: "
                f"cores [{core_thresholds[0]}, {core_thresholds[1]}], "
                f"memory [{memory_thresholds[0]}m, {memory_thresholds[1]}m].")

    history_data = get_history_data()
    history_data = history_data[history_data['status'] == 1]

    regression_model = PerformanceModel(logger=logger,
                                        core_thresholds=core_thresholds,
                                        memory_thresholds=memory_thresholds,
                                        weights=weights)
    regression_model.train(history_data)

    estimated_running_time = -1
    while True:
        params = regression_model.search_new_config(task_embedding, updated_knob_details, task_best_config)
        data = [params[knob] for knob in KNOBS]
        data.extend(task_embedding)
        predict_data = np.array(data).reshape(1, -1)
        if check_sample(params, core_thresholds, memory_thresholds):
            estimated_running_time = regression_model.predict(predict_data)[0]
            break
        else:
            logger.info(f"Config {epoch_id} for history task {update_task_id} generated failed.")
    config = params
    logger.info(f"Config {epoch_id} for history task {update_task_id} generated successfully: {config}, "
                    f"estimated running time = {estimated_running_time} ms.")

    logger.info(f"Update history of task {update_task_id} finished.")

    if weights is None:
        return config
    else:
        return config, regression_model.probabilities, regression_model.selected_index
