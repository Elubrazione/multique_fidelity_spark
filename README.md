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
使用的指令为:
```bash
python main.py --mode multi \
               --workload "TPCDS" \
               --data_size "100" \
               --type recommend-config \
               --task_id "q1_q2" \
               --model tbcnn \
               --epochs 5 \
               --task_suffix ""
```
实际使用时, 由于task_id关联到embedding vector的计算, 所以需正确设置, 对于MFTune这种使用99条query的实验略显负责, 可以将query的字符串放入new_tasks的文件中, 利用类似`run_test.sh`中的脚本进行运行.

关于历史数据以及相关query的路径配置, 都在common.py中, 注意, 默认数据路径与`workload`(诸如TPCDS), `data_size`(诸如100), `mode`(诸如multi)是相关的, 如有需要可以后续进行修改.