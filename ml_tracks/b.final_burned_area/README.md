# mesogeos Track B: Final Burned Area Prediction

## Dataset

You have two options, you can either download the dataset or generate it yourself using the [dataset_extraction.py](./dataset_extraction.py) script.

### Downloading the data

TODO: Add instructions to download the dataset.

### Generating the data

With the mesogeos datacube downloaded (see [mesogeos/README.md](../README.md)), you can generate the dataset using the [dataset_extraction.py](./dataset_extraction.py) script.

```
python dataset_extraction.py --ds_path <path_to_datacube> --gdf_path <path_to_shapefile> --save_dir <save_directory> --patch_half <half_of_patch_size_around_ignition> --days_before <days_before_ignition> --days_after <days_after_ignition>
```

## Reproducing the experiments

Set up a conda environment with the [environment.yml](./environment.yml) file. Disclaimer: this environment file is not minimal, it contains all the packages used during the development of the project. Also, it is not guaranteed to work on all platforms. It has been tested on Ubuntu 20.04 with an NVIDIA RTX 3080 GPU.

```
conda env create -f environment.yml
```

Set up wandb and fill appropriate values in the [run script](./scripts/run_experiment.sh) or disable wandb from [src/main.py](./src/main.py).


Run the script.

```
bash scripts/run_experiment.sh
```

