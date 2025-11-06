# multique_fidelity_spark
A query level multi-fidelity spark-sql tuning framework

## History Collecting
```bash
nohup env PYTHONPATH=/root/codes/multique_fidelity_spark python main.py --iter_num 200 --ws_init_num 29 --task 64u240n2 --target tpcds_100g --opt SMAC > /dev/null 2>&1 &

nohup env PYTHONPATH=/root/codes/multique_fidelity_spark python main.py --iter_num 200 --ws_init_num 29 --task 64u240n2 --target tpcds_300g --opt SMAC > /dev/null 2>&1 &
```

```bash
awk '/"objectives":/ {getline val; if (val !~ /Infinity/) c++} END {print c}' /root/codes/multique_fidelity_spark/results/tpcds_100g/64u240n2____Wnone29Tnonek-1Cnonek51sigma2.0top_ratio0.8__Sfull__s42_2025-11-04-15-02-18-141231.json
```