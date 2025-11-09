workload = 'TPCH'
data_size = 100
mode = 'single'
encoding_model = 'tbcnn'  # 'tbcnn'/'bert'/'tuneful'/'rover'/'random'
task_suffix = ''

hdfs_path = "hdfs_ip:port"
event_log_hdfs_path = "event-log-hdfs-path"

alpha = 0.1
rate_tradeoff = 0.4

# 这里是本地的MySQL用来存放历史数据的数据库对应的设置
db_user = 'root'
db_password = 'wyc685425'
db_host = '10.255.255.254'
db_port = '3306'
