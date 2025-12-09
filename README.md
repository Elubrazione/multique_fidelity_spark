# multique_fidelity_spark
A query level multi-fidelity spark-sql tuning framework

## History Collecting
```bash
nohup env PYTHONPATH=/root/codes/multique_fidelity_spark python main.py --iter_num 200 --ws_init_num 29 --task 64u240n2 --target tpcds_100g --opt SMAC > /dev/null 2>&1 &

nohup env PYTHONPATH=/root/codes/multique_fidelity_spark python main.py --iter_num 200 --ws_init_num 29 --task 64u240n2 --target tpcds_300g --opt SMAC > /dev/null 2>&1 &
```

## Run
```bash
nohup env PYTHONPATH=/root/codes/multique_fidelity_spark \
python main.py --opt MFES_SMAC \
--target tpcds_600g \
--data_dir /home/hive-testbench-hdp3/spark-queries-tpcds \
--history_dir history \
--task 64u256n3 \
--database tpcds_600g \
--resume results/tpcds_600g/MFES_SMAC/64u256n3__MFES_SMAC__Wbest_all4Treacqk3Cnonek51s2.0r0.6__Smfes_flatten__s42_2025-12-09-11-44-28-367059.json \
--transfer reacq \
--warm_start best_all \
--cp_topk 20 \
--use_flatten_scheduler \
--use_cached_model > log/all.log 2>&1 &
```