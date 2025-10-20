
from typing import Dict, Any, List
import re, os, time, csv, json
from queue import Queue
import threading
import numpy as np
from datetime import datetime
from openbox import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from ConfigSpace import Configuration

from utils.tune_ssh import TuneSSH
from utils.spark import convert_to_spark_params, custom_sort
from config import ENV_SPARK_SQL_PATH, ENV_YARN_PATH, DATABASE, DATA_DIR, RESULT_DIR, \
    LIST_SPARK_NODES, LIST_SPARK_SERVER, LIST_SPARK_USERNAME, LIST_SPARK_PASSWORD, \
    SPARK_NODES, SPARK_SERVER_NODE, SPARK_USERNAME, SPARK_PASSWORD


class ExecutorManager:
    def __init__(self, sqls: dict, timeout: dict, config_space,
                 spark_sql=ENV_SPARK_SQL_PATH, spark_nodes=LIST_SPARK_NODES,
                 servers=LIST_SPARK_SERVER, usernames=LIST_SPARK_USERNAME, passwords=LIST_SPARK_PASSWORD,
                 enable_os_tuning=False, fidelity_database_mapping=None, fixed_sqls=None):
        self.sqls = sqls
        self.timeout = timeout
        self.config_space = config_space
        self.child_num = len(spark_nodes)
        self.executor_queue = Queue()
        self.enable_os_tuning = enable_os_tuning
        self.fidelity_database_mapping = fidelity_database_mapping or {}
        self.fixed_sqls = fixed_sqls
        
        # Global setting for csv_file and csv_writer
        self.csv_path = os.path.join(RESULT_DIR, datetime.now().strftime('%Y%m%d-%H%M%S') + ".csv")
        self.csv_file = open(self.csv_path, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.all_queries = sorted(set(sum(sqls.values(), [])), key=lambda x: custom_sort(x))
        self.header = ['resource', 'query_time', 'elapsed_time', 'overhead'] + \
                      [f'qt_{q}' for q in self.all_queries] + \
                      [f'et_{q}' for q in self.all_queries] + \
                      [f'{k.name}_param' for k in config_space.get_hyperparameters()]

        if os.stat(self.csv_path).st_size == 0:
            self.csv_writer.writerow(self.header)

        self.write_queue = Queue()
        self.write_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.write_thread.start()
        self.closed = False

        self._init_child_executor(spark_sql, spark_nodes, servers, usernames, passwords)
        
        for item in sqls[round(float(1), 5)]:
            logger.info("[timeout]: %s, %f" % (item, timeout.get(item, np.inf)))

    def _init_child_executor(self, spark_sql, spark_nodes, servers, usernames, passwords):
        self.executors = []
        for idx in range(self.child_num):
            self.executor_queue.put(idx)
            self.executors.append(
                SparkTPCDSExecutor(
                    sqls=self.sqls, timeout=self.timeout,
                    write_queue=self.write_queue, all_queries=self.all_queries,
                    spark_sql=spark_sql, spark_nodes=spark_nodes[idx],
                    server=servers[idx], username=usernames[idx], password=passwords[idx],
                    enable_os_tuning=self.enable_os_tuning,
                    fidelity_database_mapping=self.fidelity_database_mapping,
                    fixed_sqls=self.fixed_sqls
                )
            )
            
    def _writer_loop(self):
        while True:
            row_dict = self.write_queue.get()
            if row_dict is None:
                break
            row = [row_dict.get(h, 0) for h in self.header]
            self.csv_writer.writerow(row)
            self.csv_file.flush()
            self.write_queue.task_done()

    def __call__(self, config, resource_ratio, res_dir):
        idx = self.executor_queue.get()  # 阻塞直到有空闲 executor
        logger.info(f"Got free executor: {idx}")

        result_queue = Queue()

        def run():
            try:
                result = self.executors[idx](config, resource_ratio, res_dir)
                result_queue.put(result)
                logger.info(f"[Executor {idx}] Result: {result}")
            finally:
                self.executor_queue.put(idx)  # 标记为“空闲”
                logger.info(f"[Executor {idx}] Marked as free again.")

        thread = threading.Thread(target=run)
        thread.start()

        result = result_queue.get()  # 等待结果
        thread.join()
        return result
    
    def __del__(self):
        self.close()

    def close(self):
        if self.closed:
            return
        try:
            self.write_queue.put(None)
            self.write_thread.join()
        finally:
            try:
                self.csv_file.close()
            finally:
                self.closed = True
                logger.info("ExecutorManager closed and CSV file saved.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class SparkTPCDSExecutor:
    def __init__(self, sqls: dict, timeout: dict, write_queue: Queue, all_queries=None,
                 suffix_type='log', spark_sql=ENV_SPARK_SQL_PATH, spark_nodes=SPARK_NODES,
                 server=SPARK_SERVER_NODE, username=SPARK_USERNAME, password=SPARK_PASSWORD,
                 enable_os_tuning=False, config_space=None,
                 fidelity_database_mapping=None, fixed_sqls=None):
        self.sqls = sqls
        self.timeout = timeout
        
        self.spark_sql = spark_sql
        self.suffix_type = suffix_type
        self.spark_nodes = spark_nodes
        self.server = server
        self.username = username
        self.password = password
        self.enable_os_tuning = enable_os_tuning
        
        self.write_queue = write_queue
        self.tune_ssh = TuneSSH(server_ip=server, server_user=username, server_passwd=password)
        self.all_queries = all_queries
        self._node_ssh_cache = {}   # cache for per-node ssh connections when OS tuning is disabled
        
        # Map fidelity to database
        self.fidelity_database_mapping = fidelity_database_mapping or {}
        
        # fixed SQL set (when using database fidelity)
        self.fixed_sqls = fixed_sqls if fixed_sqls is not None else self.get_sqls_by_fidelity(resource=1.0)
        
        self.all_os_executors = {}
        if self.enable_os_tuning:
            main_executor = OsExecutor(tune_ssh=self.tune_ssh)
            self.all_os_executors[self.server] = main_executor
            logger.info(f"[{self.server}] OS tuning enabled, OsExecutor initialized for main server")
            
            for node in self.spark_nodes:
                if node != self.server:
                    node_ssh = TuneSSH(server_ip=node, server_user=self.username, server_passwd=self.password)
                    node_executor = OsExecutor(tune_ssh=node_ssh)
                    self.all_os_executors[node] = node_executor
                    logger.info(f"[{self.server}] OsExecutor initialized for Spark node: {node}")
            
            logger.info(f"[{self.server}] Total {len(self.all_os_executors)} nodes ready for OS parameter tuning")


    def __call__(self, config, resource_ratio, res_dir):
        return self.run_spark_job(res_dir, config, resource_ratio)
    
    def get_sqls_by_fidelity(self, resource, use_delta=False):
        if self.fidelity_database_mapping and resource in self.fidelity_database_mapping:
            logger.info(f"[{self.server}] Using fixed SQL set for fidelity {resource}: {len(self.fixed_sqls)} queries")
            return self.fixed_sqls
        assert resource in self.sqls.keys()
        original_sqls = self.sqls[resource]
        if use_delta:
            for k, v in self.sqls.items():
                if k == resource: break
                original_sqls = [i for i in original_sqls if i not in v]
        return original_sqls
    
    def get_database_by_fidelity(self, fidelity):
        if self.fidelity_database_mapping and fidelity in self.fidelity_database_mapping:
            database = self.fidelity_database_mapping[fidelity]
            logger.info(f"[{self.server}] Using mapped database for fidelity {fidelity}: {database}")
            return database
        
        logger.info(f"[{self.server}] Using default database for fidelity {fidelity}: {DATABASE}")
        return DATABASE

    def run_spark_job(self, res_dir, config: dict, resource):
        logger.info(res_dir)
        start_time = time.time()
        total_time = 0
        qtime_details = {}

        logger.info("Get fidelity: " + str(resource))
        logger.debug("Get configuration: " + str(config))
        queries = self.get_sqls_by_fidelity(resource=resource)
        self.tune_ssh.exec_command(f"mkdir -p {res_dir}", id='server')

        self.clear_cache_on_remote()

        os_params = {}
        if self.all_os_executors:
            any_executor = next(iter(self.all_os_executors.values()))
            os_params = any_executor.extract_os_params(config)
        logger.info(f"[{self.server}] Separated parameters - OS: {len(os_params)}")
        
        if os_params:
            logger.info(f"[{self.server}] Applying OS parameters: {os_params}")
            self._apply_os_params_to_all_nodes(os_params)
        else:
            logger.info(f"[{self.server}] No OS parameters found in configuration")

        spark_params = {k: v for k, v in config.get_dictionary().items() if k not in os_params}
        logger.info(f"[{self.server}] Separated parameters - Spark: {len(spark_params)}")

        elapsed_time = {sql: np.Inf for sql in queries}
        total_status = True
        executed_queries = []  # 记录实际执行的查询
        
        for sql in queries:
            cur_start_time = time.time()
            status = self._run_remote_job(sql=sql, config=spark_params, res_dir=res_dir, fidelity=resource)
            elapsed_time[sql] = time.time() - cur_start_time
            executed_queries.append(sql)
            if status != "success":
                total_status = False
                break

        total_time, qtime_details = self.parse_spark_log_on_remote(
            remote_log_dir=res_dir,
            queries=executed_queries,
            status=total_status
        )
        elapsed_total_time = sum([v for v in elapsed_time.values()])
        logger.info(f"[{self.server}] Total Spark Time: {total_time}")
        logger.info(f"[{self.server}] Detailed Query Times: {qtime_details}")
        logger.info(f"[{self.server}] Detailed Elapsed Times: {elapsed_time}")
        results = self.build_ret_dict(float(total_time), start_time)
        # logger.info("Build dict results: " + str(results))
        
        row_dict = {
            'resource': resource,
            'query_time': total_time,
            'elapsed_time': elapsed_total_time,
            'overhead': elapsed_total_time - total_time
        }
        for sql in self.all_queries:
            row_dict[f'qt_{sql}'] = qtime_details.get(sql, 0)
            row_dict[f'et_{sql}'] = elapsed_time.get(sql, 0)
        for k in config:
            row_dict[f'{k}_param'] = config.get(k)
        
        if self.write_queue is not None:
            self.write_queue.put(row_dict)

        return results

    def _run_remote_job(self, sql, config, res_dir, fidelity=None):
        database = self.get_database_by_fidelity(fidelity) if fidelity is not None else DATABASE
        
        spark_cmd = [
            self.spark_sql,
            "--master", "yarn",
            "--database", database,
            *convert_to_spark_params(config),
            "-f", f"{DATA_DIR}/{sql}.sql"
        ]
        remote_log_path = f"{res_dir}/{sql}.log"
        full_cmd = f"{' '.join(spark_cmd)} > {remote_log_path} 2>&1"
        logger.info(f"[{self.server}] Running Spark command: {full_cmd}")
        timeout_sec = self.timeout.get(sql, None)
        return self._execute_command(full_cmd, timeout_sec=timeout_sec)["status"]

    def parse_spark_log_on_remote(self, remote_log_dir, queries, status):
        results = {}
        time_pattern = re.compile(r'Time taken: ([\d.]+) seconds')
        cost = 0

        try:
            for i in queries:
                results[i] = np.Inf
                remote_path = os.path.join(remote_log_dir, f"{i}.{self.suffix_type}")

                result, stderr = self.tune_ssh.exec_command(f"cat {remote_path}", 'server')
                if result is None:
                    logger.error(f"[Error reading log {remote_path}] {stderr}")
                    continue
                lines = result.splitlines()
                for line in lines:
                    time_match = time_pattern.search(line)
                    if time_match:
                        execution_time = float(time_match.group(1))
                        results[i] = execution_time
                cost += results[i]
        except Exception as e:
            logger.error(f"[parse_spark_log] Error reading remote logs: {e}")
            return float("inf"), {}

        if not status:
            cost = np.Inf
        sorted_results = dict(sorted(results.items(), key=lambda x: custom_sort(x[0])))
        return cost, sorted_results

    def _execute_command(self, full_cmd, timeout_sec=None):
        result = {"status": "error"}

        def target():
            nonlocal result
            try:
                _, return_code = self.tune_ssh.exec_command(full_cmd, 'server')
                if return_code != 0:
                    logger.error(f"[{self.server}]: Error executing command: {full_cmd}")
                    result = {"status": "fail"}
                else:
                    result = {"status": "success"}
            except Exception as e:
                logger.error({"status": "error", "error": str(e)})
                result = {"status": "error"}

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=timeout_sec)

        if thread.is_alive():
            logger.warning(f"[{self.server}]: Command timed out: {full_cmd}, killing it...")
            self._kill_all_running_yarn_apps()
            return {"status": "timeout"}

        if result["status"] == "error":
            logger.warning(f"[{self.server}]: Execution failed with error. Checking for orphaned Spark jobs to kill...")
            self._kill_all_running_yarn_apps()

        return result


    def _kill_all_running_yarn_apps(self, max_wait_seconds=30):
        logger.warning(f"Command timed out, killing it...")
        kill_cmd = f"""for app in $({ENV_YARN_PATH} application -list | grep "RUNNING" | awk '{{print $1}}'); do {ENV_YARN_PATH} application -kill $app; done"""
        _, return_code = self.tune_ssh.exec_command(kill_cmd, 'server')
        if not return_code:
            logger.info("Successfully killed timeout spark-sql.")
        else:
            logger.warn("Error while killing timeout spark-sql!!!!!!!!!!!!!!")

    def build_ret_dict(self, perf, start_time):
        result = {
            'result': {'objective': perf},
            'timeout': not np.isfinite(perf), 'traceback': None
        }
        result['elapsed_time'] = time.time() - start_time
        return  result
    
    def clear_cache_on_remote(self):
        full_cmd = "echo 3 > /proc/sys/vm/drop_caches"
        # Clear cache on the server and all worker nodes
        # main server
        result, return_code = self.tune_ssh.exec_command(full_cmd, 'server')
        if return_code != 0:
            logger.error(f"[{self.server}] Error executing command: {full_cmd}")
        # other nodes (if OS tuning enabled, we already have per-node executors; otherwise reuse server creds)
        for node in self.spark_nodes:
            if node == self.server:
                continue
            try:
                if node in self.all_os_executors:
                    node_ssh = self.all_os_executors[node].tune_ssh
                else:
                    # lazily create and cache when OS tuning is disabled
                    if node not in self._node_ssh_cache:
                        self._node_ssh_cache[node] = TuneSSH(
                            server_ip=node, server_user=self.username, server_passwd=self.password
                        )
                    node_ssh = self._node_ssh_cache[node]
                _, rc = node_ssh.exec_command(full_cmd, 'server')
                if rc != 0:
                    logger.error(f"[{node}] Error executing command: {full_cmd}")
            except Exception as e:
                logger.error(f"[{node}] Exception clearing caches: {e}")

    def _apply_os_params_to_all_nodes(self, os_params):
        if not self.all_os_executors:
            logger.warning(f"[{self.server}] No OS executors available")
            return

        logger.info(f"[{self.server}] Applying OS parameters to {len(self.all_os_executors)} nodes in parallel")
        
        results = {}
        failed_nodes = []
        successful_nodes = []
        
        with ThreadPoolExecutor(max_workers=min(len(self.all_os_executors), 8)) as executor:
            future_to_node = {}
            for node_name, os_executor in self.all_os_executors.items():
                logger.debug(f"[{self.server}] Submitting OS parameter application task for node: {node_name}")
                future = executor.submit(self._apply_os_params_to_single_node, 
                                       node_name, os_executor, os_params)
                future_to_node[future] = node_name

            for future in as_completed(future_to_node):
                node_name = future_to_node[future]
                try:
                    success = future.result()
                    results[node_name] = success
                    if success:
                        successful_nodes.append(node_name)
                        logger.info(f"[{self.server}] OS parameters applied successfully to node: {node_name}")
                    else:
                        failed_nodes.append(node_name)
                        logger.warning(f"[{self.server}] Failed to apply some OS parameters to node: {node_name}")
                except Exception as e:
                    failed_nodes.append(node_name)
                    logger.error(f"[{self.server}] Exception applying OS parameters to node {node_name}: {e}")
                    results[node_name] = False

        successful_count = len(successful_nodes)
        total_nodes = len(results)
        logger.info(f"[{self.server}] OS parameter application completed: {successful_count}/{total_nodes} nodes successful")
        
        if successful_nodes:
            logger.info(f"[{self.server}] Successful nodes: {', '.join(successful_nodes)}")
        if failed_nodes:
            logger.warning(f"[{self.server}] Failed nodes: {', '.join(failed_nodes)}")
        
        return successful_count == total_nodes
    
    def _apply_os_params_to_single_node(self, node_name, os_executor, os_params):
        try: 
            success = os_executor.apply_os_params(os_params)
            
            if success:
                logger.debug(f"[{self.server}] OS parameter application completed successfully for node: {node_name}")
            else:
                logger.warning(f"[{self.server}] OS parameter application failed for node: {node_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"[{self.server}] Exception in OS parameter application for node {node_name}: {e}")
            import traceback
            logger.debug(f"[{self.server}] Full traceback for node {node_name}: {traceback.format_exc()}")
            return False


class OsExecutor:    
    def __init__(self, tune_ssh, os_params_config_path: str = "config_space/os/os_params.json"):
        self.tune_ssh = tune_ssh
        self.os_params_config_path = os_params_config_path
        self.os_params_config = self._load_os_params_config()
        self.os_param_paths = self._init_os_param_paths()
        self.os_param_commands = self._init_os_param_commands()

        self.devices = self._detect_disk_devices()
        self.interfaces = self._detect_network_interfaces()
        logger.info(f"[{self.tune_ssh.server_ip}] Remote node detected disk devices: {self.devices}")
        logger.info(f"[{self.tune_ssh.server_ip}] Remote node detected network interfaces: {self.interfaces}")
        
    def _load_os_params_config(self) -> Dict[str, Any]:
        try:
            with open(self.os_params_config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"OS parameter configuration file not found: {self.os_params_config_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"OS parameter configuration file format error: {e}")
            return {}
    
    def _init_os_param_paths(self) -> Dict[str, str]:
        return {
            "read_ahead_kb": "/sys/block/{device}/queue/read_ahead_kb",
            "max_sectors_kb": "/sys/block/{device}/queue/max_sectors_kb",
            "nr_requests": "/sys/block/{device}/queue/nr_requests",
            "scheduler": "/sys/block/{device}/queue/scheduler",
            "vm_swappiness": "vm.swappiness",
            "vm_dirty_ratio": "vm.dirty_ratio",
            "combined_queues": "{iface}",
            "rx_ring_buffer": "{iface}",
            "tx_ring_buffer": "{iface}"
        }
    
    def _init_os_param_commands(self) -> Dict[str, str]:
        return {
            "read_ahead_kb": "echo {value} > /sys/block/{device}/queue/read_ahead_kb",
            "max_sectors_kb": "echo {value} > /sys/block/{device}/queue/max_sectors_kb",
            "nr_requests": "echo {value} > /sys/block/{device}/queue/nr_requests",
            "scheduler": "echo {value} > /sys/block/{device}/queue/scheduler",
            "vm_swappiness": "sysctl -w vm.swappiness={value}",
            "vm_dirty_ratio": "sysctl -w vm.dirty_ratio={value}",
            "combined_queues": "ethtool -L {iface} combined {value}",
            "rx_ring_buffer": "ethtool -G {iface} rx {value}",
            "tx_ring_buffer": "ethtool -G {iface} tx {value}"
        }
    
    def extract_os_params(self, config: Configuration) -> Dict[str, Any]:
        config_dict = config.get_dictionary()
        os_params = {}
        
        for param_name in self.os_params_config.keys():
            if param_name in config_dict:
                os_params[param_name] = config_dict[param_name]
        return os_params
    
    def apply_os_params(self, os_params: Dict[str, Any]) -> bool:
        if not os_params:
            logger.warning(f"[{self.tune_ssh.server_ip}] No OS parameters need to be set")
            return True
         
        success_count = 0
        total_count = len(os_params)

        disk_params = {}
        vm_params = {}
        network_params = {}
        
        for param_name, param_value in os_params.items():
            if param_name in ["read_ahead_kb", "max_sectors_kb", "nr_requests", "scheduler"]:
                disk_params[param_name] = param_value
            elif param_name in ["vm_swappiness", "vm_dirty_ratio"]:
                vm_params[param_name] = param_value
            elif param_name in ["combined_queues", "rx_ring_buffer", "tx_ring_buffer"]:
                network_params[param_name] = param_value
            else:
                logger.warning(f"[{self.tune_ssh.server_ip}] Unknown OS parameter: {param_name}")
                continue
        
        for param_name, param_value in disk_params.items():
            try:
                device_success_count = 0
                for device in self.devices:
                    if self._set_disk_param(param_name, param_value, device):
                        logger.debug(f"[{self.tune_ssh.server_ip}] Successfully set disk parameter {param_name} = {param_value} on device {device}")
                        device_success_count += 1
                    else:
                        logger.warning(f"[{self.tune_ssh.server_ip}] Failed to set disk parameter {param_name} = {param_value} on device {device}")
                
                if device_success_count > 0:
                    if device_success_count == len(self.devices):
                        logger.debug(f"[{self.tune_ssh.server_ip}] Disk parameter {param_name} successfully set on all {len(self.devices)} devices")
                    else:
                        logger.info(f"[{self.tune_ssh.server_ip}] Disk parameter {param_name} partially set on {device_success_count}/{len(self.devices)} devices")
                    success_count += 1
                else:
                    logger.error(f"[{self.tune_ssh.server_ip}] Failed to set disk parameter {param_name} = {param_value} on all devices")
                    
            except Exception as e:
                logger.error(f"[{self.tune_ssh.server_ip}] Error setting disk parameter {param_name}: {e}")
        
        for param_name, param_value in vm_params.items():
            try:
                if self._set_vm_param(param_name, param_value):
                    logger.debug(f"[{self.tune_ssh.server_ip}] Successfully set VM parameter {param_name} = {param_value}")
                    success_count += 1
                else:
                    logger.error(f"[{self.tune_ssh.server_ip}] Failed to set VM parameter {param_name} = {param_value}")
                    
            except Exception as e:
                logger.error(f"[{self.tune_ssh.server_ip}] Error setting VM parameter {param_name}: {e}")
        
        for param_name, param_value in network_params.items():
            try:
                if self._set_network_param(param_name, param_value, self.interfaces):
                    logger.debug(f"[{self.tune_ssh.server_ip}] Successfully set network parameter {param_name} = {param_value}")
                    success_count += 1
                else:
                    logger.error(f"[{self.tune_ssh.server_ip}] Failed to set network parameter {param_name} = {param_value}")
                    
            except Exception as e:
                logger.error(f"[{self.tune_ssh.server_ip}] Error setting network parameter {param_name}: {e}")
        
        logger.info(f"[{self.tune_ssh.server_ip}] Remote OS parameters set completed: {success_count}/{total_count} successfully")
        return success_count == total_count
    
    def _detect_disk_devices(self, use_nvme: bool = False) -> List[str]:
        """Detect disk devices"""
        try:
            cmd = "ls /sys/block/ | grep -E '^sd[a-z]+$'"
            result, return_code = self.tune_ssh.exec_command(cmd, 'server')
            if return_code == 0 and result:
                devices = [device for device in result.strip().split('\n') if device]
                if devices:
                    return devices
            
            if use_nvme:
                cmd = "ls /sys/block/ | grep -E '^(nvme|vd|hd)[a-z0-9]+$'"
                result, return_code = self.tune_ssh.exec_command(cmd, 'server')
                if return_code == 0 and result:
                    devices = [device for device in result.strip().split('\n') if device]
                    if devices:
                        return devices
            
            logger.warning("No disk devices detected")
            return []
        except Exception as e:
            logger.warning(f"Failed to detect disk devices: {e}")
            return []
    
    def _detect_network_interfaces(self) -> List[str]:
        """Detect network interfaces"""
        try:
            cmd = "ip link show | grep -E '^[0-9]+:' | cut -d: -f2 | tr -d ' '"
            result, return_code = self.tune_ssh.exec_command(cmd, 'server')
            if return_code == 0 and result:
                interfaces = [iface for iface in result.strip().split('\n') if iface and iface != 'lo']
                if interfaces:
                    return interfaces
            
            logger.warning("No network interfaces detected")
            return []
        except Exception as e:
            logger.warning(f"Failed to detect network interfaces: {e}")
            return []
    
    def _set_disk_param(self, param_name: str, param_value: Any, device: str) -> bool:
        try:
            cmd = self.os_param_commands[param_name].format(value=param_value, device=device)
            result, return_code = self.tune_ssh.exec_command(cmd, 'server')
            
            if return_code == 0:
                verify_result = self._verify_param_setting(param_name, param_value, device=device)
                if not verify_result:
                    try:
                        current_cmd = f"cat /sys/block/{device}/queue/{param_name}"
                        current_result, current_return_code = self.tune_ssh.exec_command(current_cmd, 'server')
                        if current_return_code == 0:
                            logger.warning(f"[{self.tune_ssh.server_ip}] Verification failed for {param_name} on {device}: expected={param_value}, actual={current_result.strip()}")
                        else:
                            logger.warning(f"[{self.tune_ssh.server_ip}] Verification failed for {param_name} on {device}: expected={param_value}, failed to read actual value")
                    except Exception as e:
                        logger.warning(f"[{self.tune_ssh.server_ip}] Verification failed for {param_name} on {device}: expected={param_value}, error reading actual value: {e}")
                return verify_result
            else:
                logger.error(f"[{self.tune_ssh.server_ip}] Failed to set remote disk parameter: {cmd}, error: {result}")
                try:
                    current_cmd = f"cat /sys/block/{device}/queue/{param_name}"
                    current_result, current_return_code = self.tune_ssh.exec_command(current_cmd, 'server')
                    if current_return_code == 0:
                        logger.error(f"[{self.tune_ssh.server_ip}] Current value of {param_name} on {device}: {current_result.strip()}")
                    else:
                        logger.error(f"[{self.tune_ssh.server_ip}] Failed to read current value of {param_name} on {device}")
                except Exception as e:
                    logger.error(f"[{self.tune_ssh.server_ip}] Error reading current value: {e}")
                return False
                
        except Exception as e:
            logger.error(f"[{self.tune_ssh.server_ip}] Failed to set remote disk parameter {param_name}: {e}")
            return False
    
    def _set_vm_param(self, param_name: str, param_value: Any) -> bool:
        try:
            cmd = self.os_param_commands[param_name].format(value=param_value)
            result, return_code = self.tune_ssh.exec_command(cmd, 'server')
            
            if return_code == 0:
                verify_result = self._verify_param_setting(param_name, param_value)
                if not verify_result:
                    try:
                        current_cmd = f"sysctl {param_name}"
                        current_result, current_return_code = self.tune_ssh.exec_command(current_cmd, 'server')
                        if current_return_code == 0:
                            logger.warning(f"[{self.tune_ssh.server_ip}] Verification failed for VM parameter {param_name}: expected={param_value}, actual={current_result.strip()}")
                        else:
                            logger.warning(f"[{self.tune_ssh.server_ip}] Verification failed for VM parameter {param_name}: expected={param_value}, failed to read actual value")
                    except Exception as e:
                        logger.warning(f"[{self.tune_ssh.server_ip}] Verification failed for VM parameter {param_name}: expected={param_value}, error reading actual value: {e}")
                return verify_result
            else:
                logger.error(f"[{self.tune_ssh.server_ip}] Failed to set remote VM parameter: {cmd}, error: {result}")
                try:
                    current_cmd = f"sysctl {param_name}"
                    current_result, current_return_code = self.tune_ssh.exec_command(current_cmd, 'server')
                    if current_return_code == 0:
                        logger.error(f"[{self.tune_ssh.server_ip}] Current value of VM parameter {param_name}: {current_result.strip()}")
                    else:
                        logger.error(f"[{self.tune_ssh.server_ip}] Failed to read current value of VM parameter {param_name}")
                except Exception as e:
                    logger.error(f"[{self.tune_ssh.server_ip}] Error reading current VM parameter value: {e}")
                return False
                
        except Exception as e:
            logger.error(f"[{self.tune_ssh.server_ip}] Failed to set remote VM parameter {param_name}: {e}")
            return False
    
    def _set_network_param(self, param_name: str, param_value: Any, interfaces: List[str]) -> bool:
        try:
            success_count = 0
            
            for iface in interfaces:
                check_cmd = f"ip link show {iface}"
                _, return_code = self.tune_ssh.exec_command(check_cmd, 'server')
                if return_code != 0:
                    logger.warning(f"[{self.tune_ssh.server_ip}] Remote network interface {iface} does not exist, skipping")
                    continue
                
                if param_name in ["combined_queues", "rx_ring_buffer", "tx_ring_buffer"]:
                    cmd = self.os_param_commands[param_name].format(iface=iface, value=param_value)
                else:
                    logger.error(f"[{self.tune_ssh.server_ip}] Unknown network parameter: {param_name}")
                    continue
                
                result, return_code = self.tune_ssh.exec_command(cmd, 'server')
                if return_code == 0:
                    if self._verify_param_setting(param_name, param_value, iface=iface):
                        logger.debug(f"[{self.tune_ssh.server_ip}] Successfully set {iface} {param_name} = {param_value}")
                        success_count += 1
                    else:
                        logger.warning(f"[{self.tune_ssh.server_ip}] Failed to verify {iface} {param_name} = {param_value}")
                else:
                    logger.warning(f"[{self.tune_ssh.server_ip}] Failed to set {iface} {param_name}: {result}")
            return success_count > 0 
            
        except Exception as e:
            logger.error(f"[{self.tune_ssh.server_ip}] Failed to set remote network parameter {param_name}: {e}")
            return False
    
    def _get_param_value(self, param_name: str, device: str = None, iface: str = None) -> Any:
        """Get current parameter value"""
        try:
            param_path = self.os_param_paths[param_name]
            
            if param_name in ["vm_swappiness", "vm_dirty_ratio"]:
                # VM parameters => use sysctl command
                cmd = f"sysctl {param_path}"
                result, return_code = self.tune_ssh.exec_command(cmd, 'server')
                if return_code == 0 and result:
                    # vm.swappiness = 60
                    lines = result.strip().split('\n')
                    for line in lines:
                        if '=' in line:
                            key, val = line.split('=', 1)
                            if param_path in key.strip():
                                return val.strip()
                return None
                
            elif param_name in ["combined_queues", "rx_ring_buffer", "tx_ring_buffer"]:
                # Network parameters => use ethtool command
                if not iface:
                    logger.error(f"[{self.tune_ssh.server_ip}] Network parameter {param_name} requires iface parameter")
                    return None
                    
                if param_name == "combined_queues":
                    cmd = f"ethtool -l {iface}"
                else:  # rx_ring_buffer or tx_ring_buffer
                    cmd = f"ethtool -g {iface}"
                
                result, return_code = self.tune_ssh.exec_command(cmd, 'server')
                if return_code == 0 and result:
                    # Parse ethtool output
                    lines = result.strip().split('\n')
                    for line in lines:
                        if 'Current hardware settings:' in line:
                            continue
                        elif param_name == "combined_queues" and 'Combined:' in line:
                            if 'Current hardware settings:' in result:
                                # line after "Current hardware settings:"
                                current_section = False
                                for l in lines:
                                    if 'Current hardware settings:' in l:
                                        current_section = True
                                        continue
                                    if current_section and 'Combined:' in l:
                                        return l.split(':')[1].strip()
                            else:
                                return line.split(':')[1].strip()
                        elif param_name == "rx_ring_buffer" and 'RX:' in line:
                            if 'Current hardware settings:' in result:
                                # line after "Current hardware settings:"
                                current_section = False
                                for l in lines:
                                    if 'Current hardware settings:' in l:
                                        current_section = True
                                        continue
                                    if current_section and 'RX:' in l:
                                        return l.split(':')[1].strip()
                            else:
                                return line.split(':')[1].strip()
                        elif param_name == "tx_ring_buffer" and 'TX:' in line:
                            if 'Current hardware settings:' in result:
                                # line after "Current hardware settings:"
                                current_section = False
                                for l in lines:
                                    if 'Current hardware settings:' in l:
                                        current_section = True
                                        continue
                                    if current_section and 'TX:' in l:
                                        return l.split(':')[1].strip()
                            else:
                                return line.split(':')[1].strip()
                return None
                
            else:
                # Disk parameters => use cat command
                if param_name in ["read_ahead_kb", "max_sectors_kb", "nr_requests", "scheduler"]:
                    if not device:
                        logger.error(f"[{self.tune_ssh.server_ip}] Disk parameter {param_name} requires device parameter")
                        return None
                    param_path = param_path.format(device=device)
                
                cmd = f"cat {param_path}"
                result, return_code = self.tune_ssh.exec_command(cmd, 'server')
                if return_code == 0 and result:
                    return result.strip()
                return None
                
        except Exception as e:
            logger.error(f"[{self.tune_ssh.server_ip}] Failed to get parameter value: {e}")
            return None

    def _verify_param_setting(self, param_name: str, expected_value: Any, device: str = None, iface: str = None) -> bool:
        try:
            actual_value = self._get_param_value(param_name, device=device, iface=iface)
            if actual_value is None:
                return False

            if isinstance(expected_value, int):
                try:
                    actual_value = int(actual_value)
                except ValueError:
                    return False
                
                # 对于read_ahead_kb，允许系统对齐 向下对齐到4的倍数
                if param_name == "read_ahead_kb":
                    if actual_value <= expected_value and actual_value >= expected_value - 3:
                        return True
                    aligned_value = (expected_value // 4) * 4
                    if actual_value == aligned_value:
                        return True

            elif isinstance(expected_value, str):
                actual_value = str(actual_value)
                if param_name == "scheduler":
                    # format "mq-deadline [kyber] bfq none"
                    import re
                    match = re.search(r'\[([^\]]+)\]', actual_value)
                    if match:
                        actual_value = match.group(1)
                    else:
                        schedulers = actual_value.split()
                        if expected_value in schedulers:
                            actual_value = expected_value
                        else:
                            return False
            
            return actual_value == expected_value
                
        except Exception as e:
            logger.error(f"[{self.tune_ssh.server_ip}] Failed to verify parameter setting: {e}")
            return False
    
    def reset_os_params_to_default(self) -> bool:
        logger.info("Reset OS parameters to default values...")
        
        os_params = {}
        for param_name, param_config in self.os_params_config.items():
            os_params[param_name] = param_config.get("default")
        
        return self.apply_os_params(os_params)
    
    def get_current_os_params(self) -> Dict[str, Any]:
        current_params = {}
        
        for param_name in self.os_param_paths.keys():
            try:
                if param_name in ["read_ahead_kb", "max_sectors_kb", "nr_requests", "scheduler"]:
                    # get value from first device
                    device = self.devices[0] if self.devices else None
                    value = self._get_param_value(param_name, device=device)
                elif param_name in ["vm_swappiness", "vm_dirty_ratio"]:
                    value = self._get_param_value(param_name)
                elif param_name in ["combined_queues", "rx_ring_buffer", "tx_ring_buffer"]:
                    # get value from first interface
                    iface = self.interfaces[0] if self.interfaces else None
                    value = self._get_param_value(param_name, iface=iface)
                else:
                    value = None

                if value is not None:
                    try:
                        value = int(value)
                    except ValueError:
                        pass
                current_params[param_name] = value
                
            except Exception as e:
                logger.warning(f"[{self.tune_ssh.server_ip}] Failed to read parameter {param_name}: {e}")
                current_params[param_name] = None
        
        return current_params


if __name__ == "__main__":
    pass