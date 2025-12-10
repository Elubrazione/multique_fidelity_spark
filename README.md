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

实际运行实验时, 使用如下指令
```bash
nohup env PYTHONPATH=/root/codes/multique_fidelity_spark python main.py --opt rover --iter_num 200 --task rover_tpch600g_64u240n3 --target tpch_600g --save_dir rover_test --warm_start none --tl_topk 5 --compress shap > log/all.log 2>&1 &
```