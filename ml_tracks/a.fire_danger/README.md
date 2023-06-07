# mesogeos Track B: Wildfire Danger Forecasting

## Dataset

You have two options, you can either download the dataset or generate it yourself using the two notebooks [dataset_extraction/positives.csv](./dataset_extraction/positives.ipynb)
and [dataset_extraction/positives.csv](./dataset_extraction/positives.ipynb) notebooks.

### Downloading the data

TODO: Add instructions to download the dataset.

### Generating the data

With the mesogeos datacube downloaded (see [mesogeos/README.md](/README.md)), you can generate the dataset using the two notebooks [dataset-extraction/positives.ipynb](dataset-extraction/positives.ipynb)
and [dataset-extraction/negatives.ipynb](dataset-extraction/negatives.ipynb) notebooks.

To reproduce the scripts you should also download the shapefile with the burned areas and the shapefile with the Ecoregions and add the paths to the empty path templates in the notebooks..

## Reproducing the experiments

Install all the requirements from the [requirements.txt](./requirements.txt) file. Disclaimer: this environment file is not minimal, it contains all the packages used during the development of the project. Also, it is not guaranteed to work on all platforms. It has been tested on Ubuntu 20.04 with an NVIDIA RTX 3080 GPU.

```
pip install -r requirements.txt
```

To run the experiments that presented in the paper for the three models LSTM, Transformer and Gated Transformet Network, by running: 
```
- train.py --config configs/config_lstm/config_train.json 
- train.py --config configs/config_transformer/config_train.json 
- train.py --config configs/config_gtn/config_train.json
```
Before running the experiments you should add the dataset path to the "dataset_root" in the config files.

Similarly for testing the results: 
```
- test.py --config configs/config_lstm/config_test.json 
- test.py --config configs/config_transformer/config_test.json 
- test.py --config configs/config_gtn/config_test.json
```
Before running the tests you should add the dataset path to the "dataset_root" and the trained model path to the "model_path" in the config files 