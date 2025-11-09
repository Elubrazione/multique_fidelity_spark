


def fake_executor(task_id, config):
    # 返回总duration时间以及所有query执行结果的"与", 也即一个bool值表示是否全部成功
    # 这里面可以接executor的逻辑, 目前先mock
    return 10, True