# multiversePy

## Setup
1. Install requirements: `pip install -r requirements.txt`
1. Download datasets into respective folders in `/data/`
    * [Bitcoin Heist](https://archive.ics.uci.edu/ml/datasets/BitcoinHeistRansomwareAddressDataset)
    * [kdd cup](https://drive.google.com/drive/folders/1cavYoE2ocmAYlP0VIWiT6Q-JTrpnHn6T?usp=sharing)
1. Run preprocessing script: `python preprocess.py` for downloaded datasets
1. Run main program: `python main.py`

## Todo
* [x] Clean the datasets
* [x] Save clean versions of datasets
* [ ] Look into Compute Canada
* [ ] UCI data downloading 
* [ ] Scale features if too big
* [x] Create configs for datasets [(start with Huseyin's 18 datasets)](https://drive.google.com/drive/folders/1cavYoE2ocmAYlP0VIWiT6Q-JTrpnHn6T?usp=sharing)
* [x] Run two level RF analysis on the datasets
* [x] Record entropy for major poisoning levels
* [ ] Record performance (AUC, Bias, LogLoss) of the first level RF trained on test data
* [ ] Use the fewest neurons and the fewest neural network layers to reach RF performance (or use the same number of neurons and layers for all datasets and compare performance results?)
* [ ] Based on RF breaking point and NN simplicity, explain data in global terms (global explanations) or in terms of salient data points (local explanations). Both are open research problems.
* [ ] Global explanations can be managed by using functional data depth on entropy lines? Reporting breaking points in performance wrt. the poisoning rate?
* [ ] Local explanations (which data points' removal cause the biggest drop in datasets)

## Dataset Info

### Dataset Links
* [adult](https://archive.ics.uci.edu/ml/datasets/Adult)
* [bc](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
* [btc heist](https://archive.ics.uci.edu/ml/datasets/BitcoinHeistRansomwareAddressDataset)
* [Huseyin's](https://drive.google.com/drive/folders/1cavYoE2ocmAYlP0VIWiT6Q-JTrpnHn6T?usp=sharing)

### Ignored Datasets
| Dataset              | Reason             |
|----------------------|--------------------|
| haberman             | too few features   |
| temp of major cities | not classification |
| wisconsin cancer     | duplicate          |


## Other Links
* [Shapley values](https://dalex.drwhy.ai/)
* [Shapley values book](https://ema.drwhy.ai/shapley.html)
