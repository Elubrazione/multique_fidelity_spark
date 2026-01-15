# multique_fidelity_spark
A query level multi-fidelity spark-sql tuning framework

## History Collecting
```bash
nohup env PYTHONPATH=/root/codes/multique_fidelity_spark python main.py --iter_num 200 --ws_init_num 29 --task 64u240n2 --target tpcds_100g --opt SMAC > /dev/null 2>&1 &

nohup env PYTHONPATH=/root/codes/multique_fidelity_spark python main.py --iter_num 200 --ws_init_num 29 --task 64u240n2 --target tpcds_300g --opt SMAC > /dev/null 2>&1 &
```

## Run
### TPCDS
```bash
nohup env PYTHONPATH=/root/codes/multique_fidelity_spark \
python main.py \
--opt MFES_SMAC \
--target tpcds_600g \
--data_dir /home/hive-testbench-hdp3/spark-queries-tpcds \
--history_dir history \
--task 64u256n3 \
--database tpcds_600g \
--resume 64u256n3_default_config.json \
--transfer reacq \
--warm_start best_all \
--cp_topk 30 \
--R 9 \
--use_flatten_scheduler \
--use_cached_model > log/all.log 2>&1 &
```

### TPCH
```bash
nohup env PYTHONPATH=/root/codes/multique_fidelity_spark \
python main.py \
--opt MFES_SMAC \
--target ours \
--data_dir /srv/BigData/hadoop/data1/tpch-for-spark-sql/dbgen/saveSql \
--history_dir results \
--task 64u256n3 \
--database tpch_600g \
--transfer reacq \
--warm_start best_all \
--cp_topk 30 \
--R 9 \
--use_flatten_scheduler \
--use_cached_model > log/all.log 2>&1 &
```