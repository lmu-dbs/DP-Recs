# DP Recs

### Rule Mining Process
For creating rules please refer to ER-Miner implemented in [SPMF](https://www.philippe-fournier-viger.com/spmf/ERMiner.php) by Prof. Fournier-Viger.

### Simulation Process
- To successively run both simulation process simply run the script in `dp_recs.py`
- Currently the simulation of both example datasets is activated
- If one would like to mine rules with custom parameters, please adapt them in function `er_miner()`. Also adapt the path `spmf_bin_location_dir` which should refer to the spmf location.
- To adjust size of cutoff and time window, please have a look at `constants.py`

### Datasets
The folder `datasets` contains two example dataset used for experiments in the paper. These datasets are already preprocessed. As an example *BMSWebView1* is already split into training and test set. Note, that each split is performed randomly, hence, your experimental results may differ.