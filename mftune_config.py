import os
import json


class ConfigManager:
    
    def __init__(self, config_file='configs/spark.json'):
        self.config_file = config_file
        self.config = self._load_config()
        self._setup_constants()
    
    def _load_config(self):
        config_path = os.path.join(os.path.dirname(__file__), self.config_file)
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _setup_constants(self):
        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        
        self.LOG_DIR = os.path.join(self.ROOT_DIR, self.config['log_dir'])
        self.RESULT_DIR = os.path.join(self.ROOT_DIR, self.config['result_dir'])
        
        self.DATA_DIR = self.config['data_dir']
        self.DATABASE = self.config['database']
        self.SPARK_NODES = self.config['spark']['nodes']
        self.SPARK_SERVER_NODE = self.config['spark']['server_node']
        self.SPARK_USERNAME = self.config['spark']['username']
        self.SPARK_PASSWORD = self.config['spark']['password']
        
        self.LIST_SPARK_USERNAME = self.config['multi_cluster']['usernames']
        self.LIST_SPARK_PASSWORD = self.config['multi_cluster']['passwords']
        self.LIST_SPARK_SERVER = self.config['multi_cluster']['servers']
        self.LIST_SPARK_NODES = self.config['multi_cluster']['nodes']
        
        self.ENV_SPARK_SQL_PATH = self.config['env']['spark_sql_path']
        self.ENV_YARN_PATH = self.config['env']['yarn_path']
        
        self.FILE_SQL_SEGMENTATION = os.path.join(self.ROOT_DIR, self.config['files']['sql_segmentation'])
        self.FILE_TIMEOUT_CSV = os.path.join(self.ROOT_DIR, self.config['files']['timeout_csv'])
        self.FILE_TIMEOUT_LOG = os.path.join(self.ROOT_DIR, self.config['files']['timeout_log'])
        
        self.HISTORY_COMPRESS_DIR = os.path.join(self.ROOT_DIR, self.config['history']['compress_dir'])
        self.RANGE_COMPRESS_DATA = os.path.join(self.ROOT_DIR, self.config['history']['range_compress_data'])
        
        self.HUGE_SPACE_FILE = os.path.join(self.ROOT_DIR, self.config['config_space'])
        self.OS_CONFIG_SPACE_FILE = os.path.join(self.ROOT_DIR, self.config['os_config_space'])
        self.EXPERT_PARAMS_FILE = os.path.join(self.ROOT_DIR, self.config['expert_space'])


config_manager = ConfigManager()

ROOT_DIR = config_manager.ROOT_DIR
LOG_DIR = config_manager.LOG_DIR
RESULT_DIR = config_manager.RESULT_DIR
DATA_DIR = config_manager.DATA_DIR
ENV_SPARK_SQL_PATH = config_manager.ENV_SPARK_SQL_PATH
ENV_YARN_PATH = config_manager.ENV_YARN_PATH
DATABASE = config_manager.DATABASE
SPARK_NODES = config_manager.SPARK_NODES
SPARK_SERVER_NODE = config_manager.SPARK_SERVER_NODE
SPARK_USERNAME = config_manager.SPARK_USERNAME
SPARK_PASSWORD = config_manager.SPARK_PASSWORD
LIST_SPARK_USERNAME = config_manager.LIST_SPARK_USERNAME
LIST_SPARK_PASSWORD = config_manager.LIST_SPARK_PASSWORD
LIST_SPARK_SERVER = config_manager.LIST_SPARK_SERVER
LIST_SPARK_NODES = config_manager.LIST_SPARK_NODES
FILE_SQL_SEGMENTATION = config_manager.FILE_SQL_SEGMENTATION
FILE_TIMEOUT_CSV = config_manager.FILE_TIMEOUT_CSV
FILE_TIMEOUT_LOG = config_manager.FILE_TIMEOUT_LOG
HISTORY_COMPRESS_DIR = config_manager.HISTORY_COMPRESS_DIR
RANGE_COMPRESS_DATA = config_manager.RANGE_COMPRESS_DATA
HUGE_SPACE_FILE = config_manager.HUGE_SPACE_FILE
OS_CONFIG_SPACE_FILE = config_manager.OS_CONFIG_SPACE_FILE
EXPERT_PARAMS_FILE = config_manager.EXPERT_PARAMS_FILE