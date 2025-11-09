from Evaluator.mock_executor import MockExecutor
from Evaluator.executor import SparkSessionTPCDSExecutor

def fake_executor(task_id, config):
    # 返回总duration时间以及所有query执行结果的"与", 也即一个bool值表示是否全部成功
    # 这里面可以接executor的逻辑, 目前先mock
    executor = MockExecutor()
    resource_ratio = round(float(1.0), 5)
    observation_dict = executor(config, resource_ratio)
    app_succeeded = not observation_dict['timeout']
    return observation_dict['result']['objective'], app_succeeded