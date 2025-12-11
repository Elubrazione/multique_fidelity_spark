from openbox import logger
from mftune_config import DATA_DIR
import os, paramiko

def analyze_timeout_and_get_fidelity_details(file_path=None, percentile=100,
                                             ratio_list=[], round_num=5, debug=False, add_on_ratio=1.5):
    # 提取DATA_DIR下所有的文件名作为list
    file_list = [os.path.splitext(f)[0] for f in os.listdir(DATA_DIR)]
    remove_list = ['README', 'LICENSE']
    file_list = [item for item in file_list if item not in remove_list]
    fidelity_details = {round(1, 5): sorted(file_list, key=lambda x: x[1:])}
    elapsed_timeout_dicts = {key: 10000 for key in sorted(file_list, key=lambda x: x[1:])}
    return fidelity_details, elapsed_timeout_dicts

def clear_cache_on_remote(server, username = "root", password = "root"):
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(server, username=username, password=password)
        stdin, stdout, stderr = client.exec_command("echo 3 > /proc/sys/vm/drop_caches")
        error = stderr.read().decode()
        if error:
            logger.error(f"[{server}] Error: {error}")
        else:
            logger.info(f"[{server}] Cache cleared successfully.")

        stdin, stdout, stderr = client.exec_command("free -g")
        logger.info(f"[{server}] Memory status:\n{stdout.read().decode()}")

        client.close()
    except Exception as e:
        logger.error(f"[{server}] Error: {e}")
