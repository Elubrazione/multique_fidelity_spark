# multique_fidelity_spark
A query level multi-fidelity spark-sql tuning framework

## LOCAT Method
1. **QCSA (Query Configuration Sensitivity Analysis)**: Identifies configuration-sensitive queries (CSQ) and removes configuration-insensitive queries (CIQ) to create a Reduced Query Application (RQA).

2. **IICP (Identifying Important Configuration Parameters)**: Selects important configuration parameters through:
   - **CPS (Configuration Parameter Selection)**: Uses Spearman Correlation Coefficient to filter parameters
   - **CPE (Configuration Parameter Extraction)**: Uses Kernel PCA (Gaussian kernel) to extract important features

3. **DAGP (Data-size aware Gaussian Process)**: A surrogate model that considers both configuration parameters and input data size.

### Usage

```bash
python main.py --opt LOCAT --transfer reacq --test_mode --iter_num 10
```

### Configuration

LOCAT parameters can be configured in the config file:
- `n_qcsa`: Minimum number of samples for QCSA analysis (default: 20, used as min_samples)
- `n_iicp`: Minimum number of samples for IICP analysis (default: 10, used as min_samples)
- `scc_threshold`: Spearman correlation threshold for CPS (default: 0.2)
- `kpca_kernel`: Kernel type for KPCA (default: 'rbf')

## History Collecting
```bash
nohup env PYTHONPATH=/root/codes/multique_fidelity_spark python main.py --iter_num 200 --ws_init_num 29 --task 64u240n2 --target tpcds_100g --opt SMAC > /dev/null 2>&1 &

nohup env PYTHONPATH=/root/codes/multique_fidelity_spark python main.py --iter_num 200 --ws_init_num 29 --task 64u240n2 --target tpcds_300g --opt SMAC > /dev/null 2>&1 &
```