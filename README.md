# LOFTune

This repository contains the source code for our paper: **LOFTune: A Low-overhead and Flexible Approach for Spark SQL Configuration Tuning**.

# Requirements
***
- tokenizers 0.11.4
- optuna 3.5.0
- quantile-forest 1.1.3
- scikit-learn 1.0.2
- torch 1.12.1
- tree-sitter 0.20.1
- sqlglot 20.7.1
- hdfs
***

# Datasets
***
- [TPCDS(100G and 300G)](https://www.tpc.org/tpcds/)
- [TPCH(100G)](https://www.tpc.org/tpch/)
- [IMDB](http://homepages.cwi.nl/~boncz/job/imdb.tgz)

# Structure
***
- config: The parameters of the algorithm and model.
- data: Part of datasets used in the experiments.
- modules: Knowledge Base Updater, Configuration Recommender, Controller and some helper functions.
- sql_encoder: Convert sql to vector, i.e. Multi-task SQL Representation Learning.
- main.py: A complete function entrance, including all callable related interfaces.
- run_tests.sh: A shell test script that can be run directly.
- scripts and utils.py: Some commonly used helper functions.
***

# Usage
***
1. Download datasets
2. Set mode and workloads in run_tests.sh
3. Execute run_tests.sh

# Baseline复现
> 一些具体实现时的点:
> 
> 由于LOFTune中TPCH与TPCDS的任务使用的历史严格区分开, 故对于使用所有queries的本次实验, encode部分徒增计算量并无任何实际作用, 经后续思考, 决定放弃encode这一部分.

TPCH部分实验:

先前往`./configs/spark.json`中修改:
- `"database": "tpch_600g"`
- `"data_dir": "/srv/BigData/hadoop/data1/tpch-for-spark-sql/dbgen/saveSql/"`
- 修改`result_dir`和`json_file_name`共同指定最终输出的history的位置

然后执行指令:
```bash
nohup python main.py --mode multi \
               --workload "TPCH" \
               --data_size "600" \
               --type recommend-config \
               --task_id "all" \
               --model tbcnn \
               --epochs 200 \
               --task_suffix "" > logs/all.log 2>&1 &
```

---
TPCDS部分实验

先前往`./configs/spark.json`中修改
- `"database": "tpcds_600g"`
- `"data_dir": "/home/hive-testbench-hdp3/spark-queries-tpcds/"`
- 修改`result_dir`和`json_file_name`共同指定最终输出的history的位置

然后执行指令:
```bash
nohup python main.py --mode multi \
               --workload "TPCDS" \
               --data_size "600" \
               --type recommend-config \
               --task_id "all" \
               --model tbcnn \
               --epochs 200 \
               --task_suffix "" > logs/all.log 2>&1 &
```

> 附注, 上述指令中的`mode`, `workload`, `data_size`三个选项, 实际作用为历史数据定位, 实际上executor部分由于沿用了MFTune, 故实际运行的数据库等相关内容在`./configs/spark.json`中设置

关于`json`历史数据到支持LOFTune的`csv`历史数据的转化, 转化工具在`./scripts/json_to_history.py`中, 使用时记得在当前文件夹下 (模块才能正确加载), 也即`python ./scripts/json_to_history.py`, 目前已完成TPCH_600G和TPCDS_600G历史数据转化