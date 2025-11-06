import os
import sys
import argparse
import config.config
import pymysql
from sqlalchemy_utils import database_exists, create_database


def conf_check():
    from config.common import sql_base_path, loftune_db_url, cwd, db_name, config_path
    from config.encoder_config import tree_sitter_sql_lib_path

    if not os.path.exists(sql_base_path): # ?待调优的SQL查询文件所在文件夹? 例如q28.sql之类的
        print("The path of SQL statements does not exist, please specify `sql_base_path` in common.py.")
        sys.exit()

    if not os.path.exists(config_path): # Spark配置文件路径
        print(f"The path for Spark Configuration files does not exist, creating directory {config_path}.")
        os.makedirs(config_path)

    if not os.path.exists(tree_sitter_sql_lib_path): # Tree-sitter SQL语法解析器编译后的动态库路径
        print("sql.so is not found, please specify `tree_sitter_sql_lib_path` in encoder_config.py.")
        sys.exit()

    pymysql.install_as_MySQLdb()
    if not database_exists(loftune_db_url): # 这里是一个MySQL数据库 (猜测是它们存储history的地方)
        print(f"Database {db_name} does not exist, creating database according to the connection in config.py.")
        create_database(loftune_db_url)

    if not os.path.exists(f'{cwd}/data'):
        os.makedirs(f'{cwd}/data')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default="", choices=['IMDB', 'TPCDS', 'TPCH', 'JOIN', 'SCAN', 'AGGR'],
                        help="The benchmark name.")
    parser.add_argument('--data_size', type=int, default=0, help="The data size in GB.")
    parser.add_argument('--type', type=str, default='',
                        choices=['init-tuning-data', 'recommend-config',
                                 'recommend-config-no-history', 'update-history',
                                 'recommend-config-alternately'],
                        help='Decide what to do.')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'multi'])
    parser.add_argument('--model', type=str, default='tbcnn', choices=['tbcnn', 'bert', 'tuneful', 'rover', 'random']) # model上, 为什么rover会和tbcnn同级?
    parser.add_argument('--task_suffix', type=str, default='')
    parser.add_argument('--tradeoff', type=float, default=0.4)
    parser.add_argument('--task_id', type=str, default='', help='The task for operation.')
    parser.add_argument('--epochs', type=int, default=2, help='The number of sampled configs for each history task.') # 优化轮次, 与论文中max_iter相同
    parser.add_argument('--random_epochs', type=int, default=10,
                        help='The number of sampled configs for each history task.') # 仅在没有history的情况下使用, 表示前多少轮使用随机采样
    parser.add_argument('--per_round', type=int, default=1, help='The number of sampled configs for each history task.') # 完全没被使用
    opt = parser.parse_args()

    config.config.workload = opt.workload
    config.config.data_size = opt.data_size
    config.config.mode = opt.mode
    config.config.encoding_model = opt.model
    config.config.task_suffix = opt.task_suffix
    config.config.rate_tradeoff = opt.tradeoff

    conf_check()

    if opt.type == 'init-tuning-data':
        from modules.tuning_data_initializer import init_tuning_data
        init_tuning_data()

    # python main.py --type recommend-config --task_id {the task id for config recommendation}
    # 复现的时候使用这个就好
    elif opt.type == 'recommend-config':
        if opt.task_id == '':
            print("Please specify the task id.")
            sys.exit()
        from modules.controller import recommend_config_for_new_task
        recommend_config_for_new_task(opt.task_id, opt.epochs)

    elif opt.type == 'recommend-config-alternately':
        if opt.task_id == '':
            print("Please specify the task id.")
            sys.exit()
        from modules.controller import recommend_config_alternately
        recommend_config_alternately(opt.task_id, opt.epochs)

    elif opt.type == 'recommend-config-no-history':
        if opt.task_id == '':
            print("Please specify the task id.")
            sys.exit()
        from modules.controller import recommend_config_for_new_task_without_history
        recommend_config_for_new_task_without_history(opt.task_id, random_sample_epochs=opt.random_epochs,
                                                      model_sample_epochs=opt.epochs)

    # python main.py --type update-history --task_id {the task id for history update}
    #                                      --epochs {the number of sampled configs for each history task}
    elif opt.type == 'update-history':
        if opt.task_id == '':
            print("Please specify the task id.")
            sys.exit()
        from modules.controller import update_history_task
        update_history_task(opt.task_id, opt.epochs)

