# multique_fidelity_spark
A query level multi-fidelity spark-sql tuning framework

## History Collecting
```bash
nohup env PYTHONPATH=/root/codes/multique_fidelity_spark python main.py --iter_num 200 --ws_init_num 29 --task 64u240n2 --target tpcds_100g --opt SMAC > /dev/null 2>&1 &

nohup env PYTHONPATH=/root/codes/multique_fidelity_spark python main.py --iter_num 200 --ws_init_num 29 --task 64u240n2 --target tpcds_300g --opt SMAC > /dev/null 2>&1 &
```


# Tuneful example
```bash
python main.py --opt tuneful --iter_num 30 --test_mode --history_dir mock/history --save_dir tuneful_test
```

# Rover example
```bash
python main.py --opt rover --iter_num 10 --test_mode --history_dir mock/history --save_dir rover_test --warm_start none --tl_topk 5 --compress shap
```

实际运行实验时, 需先修改`configs/base.yaml`中的相关配置:
- 修改`database`为相应的数据库 (`tpch_600g`或者`tpcds_600g`)
- 修改`data_dir`为相应的query位置, 对于tpch数据库, 153位于`/srv/BigData/hadoop/data1/tpch-for-spark-sql/dbgen/saveSql`, 76位于`/srv/BigData/hadoop/data8/source_code/dbgen/saveSql`, 对于tpcds数据库, 位于`/home/hive-testbench-hdp3/spark-queries-tpcds`
- 修改target为相应的数据库 (`tpch_600g`或者`tpcds_600g`)

使用如下指令
```bash
nohup env PYTHONPATH=/root/codes/multique_fidelity_spark python main.py --opt rover --iter_num 200 --history_dir results/tpch_600g --task rover_tpch600g_64u240n3 --target tpch_600g --save_dir rover_test --warm_start none --tl_topk 5 --compress shap > log/all.log 2>&1 &
```