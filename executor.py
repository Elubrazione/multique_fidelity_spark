
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
                 fidelity_database_mapping=None, fixed_sqls=None):
        self.sqls = sqls
        self.timeout = timeout
        self.config_space = config_space
        self.child_num = len(spark_nodes)
        self.executor_queue = Queue()
        self.fidelity_database_mapping = fidelity_database_mapping or {}
        self.fixed_sqls = fixed_sqls
        
        # Global setting for csv_file and csv_writer
        if not os.path.exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)
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
                 config_space=None,
                 fidelity_database_mapping=None, fixed_sqls=None):
        self.sqls = sqls
        self.timeout = timeout
        
        self.spark_sql = spark_sql
        self.suffix_type = suffix_type
        self.spark_nodes = spark_nodes
        self.server = server
        self.username = username
        self.password = password
        
        self.write_queue = write_queue
        self.tune_ssh = TuneSSH(server_ip=server, server_user=username, server_passwd=password)
        self.all_queries = all_queries
        self._node_ssh_cache = {}
        
        self.fidelity_database_mapping = fidelity_database_mapping or {}
        self.fixed_sqls = fixed_sqls if fixed_sqls is not None else self.get_sqls_by_fidelity(resource=1.0)
        

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

        if hasattr(config, 'get_dictionary'):
            config_dict = config.get_dictionary()
        else:
            config_dict = config

        elapsed_time = {sql: np.Inf for sql in queries}
        total_status = True
        executed_queries = []  # 记录实际执行的查询
        
        for sql in queries:
            cur_start_time = time.time()
            status = self._run_remote_job(sql=sql, config=config_dict, res_dir=res_dir, fidelity=resource)
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
                
                # Extract all Time taken values and sum them
                total_execution_time = 0.0
                time_taken_count = 0
                for line in lines:
                    time_match = time_pattern.search(line)
                    if time_match:
                        execution_time = float(time_match.group(1))
                        total_execution_time += execution_time
                        time_taken_count += 1
                
                # If we found any Time taken values, use the sum; otherwise keep np.Inf
                if time_taken_count > 0:
                    results[i] = total_execution_time
                    logger.debug(f"[{self.server}] Query {i}: Found {time_taken_count} Time taken entries, total: {total_execution_time}")
                else:
                    logger.warning(f"[{self.server}] Query {i}: No Time taken entries found in log")
                
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
        _, return_code = self.tune_ssh.exec_command(full_cmd, 'server')
        if return_code != 0:
            logger.error(f"[{self.server}] Error executing command: {full_cmd}")
        for node in self.spark_nodes:
            if node == self.server:
                continue
            if node not in self._node_ssh_cache:
                self._node_ssh_cache[node] = TuneSSH(
                    server_ip=node, server_user=self.username, server_passwd=self.password
                )
            node_ssh = self._node_ssh_cache[node]
            _, rc = node_ssh.exec_command(full_cmd, 'server')
            if rc != 0:
                logger.error(f"[{node}] Error executing command: {full_cmd}")
