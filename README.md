# multique_fidelity_spark
A query level multi-fidelity spark-sql tuning framework

## History Collecting
```bash
nohup env PYTHONPATH=/root/codes/multique_fidelity_spark python main.py --iter_num 200 --ws_init_num 29 --task 64u240n2 --target tpcds_100g --opt SMAC > /dev/null 2>&1 &

nohup env PYTHONPATH=/root/codes/multique_fidelity_spark python main.py --iter_num 200 --ws_init_num 29 --task 64u240n2 --target tpcds_300g --opt SMAC > /dev/null 2>&1 &
```

# Rover example
```bash
python main.py --opt rover --iter_num 10 --test_mode --history_dir mock/history --save_dir rover_test --warm_start none --tl_topk 5 --compress shap
```

# Rover 实验运行
关于环境, 直接沿用MFTune的环境, 也即`spark-test`即可

运行tpch_600g实验时:
需先修改`configs/base.yaml`中的相关配置:
- 修改`database`为相应的数据库 (`tpch_600g`)
- 修改`data_dir`为相应的query位置, 对于tpch数据库, 153位于`/srv/BigData/hadoop/data1/tpch-for-spark-sql/dbgen/saveSql`, 76位于`/srv/BigData/hadoop/data8/source_code/dbgen/saveSql`
- 修改target为相应的数据库 (`tpch_600g`)

最后, 在`spark-test`的conda环境下, 使用如下指令
```bash
nohup env PYTHONPATH=/root/codes/multique_fidelity_spark python main.py --opt rover --iter_num 200 --history_dir results/tpch_600g --task rover_tpch600g_64u240n3 --target tpch_600g --save_dir rover_test --warm_start none --tl_topk 5 --compress shap > log/all.log 2>&1 &
```

运行tpcds_600g实验时:
需先修改`configs/base.yaml`中的相关配置:
- 修改`database`为相应的数据库 (`tpcds_600g`)
- 修改`data_dir`为相应的query位置, 对于tpcds数据库, 位于`/home/hive-testbench-hdp3/spark-queries-tpcds`
- 修改target为相应的数据库 (`tpcds_600g`)

最后, 在`spark-test`的conda环境下, 使用如下指令
```bash
nohup env PYTHONPATH=/root/codes/multique_fidelity_spark python main.py --opt rover --iter_num 200 --history_dir results/tpcds_600g --task rover_tpcds600g_64u240n3 --target tpcds_600g --save_dir rover_test --warm_start none --tl_topk 5 --compress shap > log/all.log 2>&1 &
```


> 补充说明, 最终的json历史数据将保存于`<save_dir>/<target>/<opt>`下, 命名前缀为`<task>`, 日志将位于`log/<target>/<opt>`下, 也可以在`configs/base.yaml`中指定.

> 另外, 由于Rover计算similarity时是沿用了`taskManager`中的点积计算方式, 故使用`json`历史数据时, 若出现了对不齐的现象, 可以考虑将json文件中`meta_feature`相关内容中, 指代cores数/内存/节点数的三个特征手动删除


python main.py --opt tuneful --iter_num 30 --test_mode --history_dir mock/history --save_dir tuneful_test

python main.py --opt toptune --iter_num 30 --test_mode --history_dir mock/history --save_dir tuneful_test

python main.py --opt rover --iter_num 30 --test_mode --history_dir mock/history --save_dir tuneful_test


# original 场景
nohup python main.py --opt tuneful --iter_num 200 --history_dir results/tpch_600g --task tuneful_tpch600g_64u240n3 --target tpch_600g --save_dir tuneful_test > ./log.log 2>&1 &

nohup python main.py --opt toptune --iter_num 300 --history_dir results/tpch_600g --task toptune_tpch600g_64u240n3 --target tpch_600g --save_dir exp_results > ./log.log 2>&1 &


nohup python main.py --opt toptune --iter_num 300 --history_dir results/tpch_600g --task toptune_tpch600g_64u240n3 --target tpch_600g --save_dir exp_results  --resume /root/codes/multique_fidelity_spark/exp_results/tpch_600g/toptune/toptune_tpch600g_64u240n3__toptune__Sfull__s42_2025-12-16-22-00-23-124592.json > ./log.log 2>&1 &

awk '/"objectives":/ {getline val; if (val !~ /Infinity/) c++} END {print c}' 

# cross benchmark 场景
nohup python main.py --opt rover --iter_num 200 --history_dir results/tpcds_600g --task rover_tpch600g_64u240n3 --target tpch_600g_crossbench --save_dir exp_results --resume /root/codes/multique_fidelity_spark/exp_results/tpch_600g_crossbench/rover/rover_tpch600g_64u240n3__rover__Sfull__s42_2025-12-22-12-53-55-486721.json> ./log.log 2>&1 &

nohup python main.py --opt tuneful --iter_num 200 --history_dir results/tpcds_600g --task tuneful_tpch600g_64u240n3 --target tpch_600g_crossbench --save_dir exp_results --resume /root/codes/multique_fidelity_spark/exp_results/tpch_600g_crossbench/tuneful/tuneful_tpch600g_64u240n3__tuneful__Sfull__s42_2025-12-25-05-20-28-157471.json > ./log.log 2>&1 &