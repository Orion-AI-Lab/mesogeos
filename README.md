# Mesogeos: A multi-purpose dataset for data-driven wildfire modeling in the Mediterranean

🆕 2023-09: Accepted at [Neurips 2023 Datasets and Benchmarks Track](https://openreview.net/group?id=NeurIPS.cc/2023/Track/Datasets_and_Benchmarks)

This is the official code repository of the mesogeos dataset. 

[Pre-print](https://arxiv.org/abs/2306.05144) describing the paper.

This repo contains code for the following:
* Creation of the Mesogeos datacube.
* Extraction of machine learning datasets for different tracks.
* Training and evaluation machine learning models for these tracks.

**Authors**: *Spyros Kondylatos (1, 2), Ioannis Prapas (1, 2), Gustau Camps-Valls (2), Ioannis Papoutsis (1)*

*(1) Orion Lab, IAASARS, National Observatory of Athens*

*(2) Image & Signal Processing Group, Universitat de València*

## Table of Contents

- [Downloading the data](#downloading-the-data)
- [Datacube Generation](#datacube-generation)
- [Machine Learning Tracks](#machine-learning-tracks)
 - [Track A: Wildfire Danger Forecasting](#track-a-wildfire-danger-forecasting)
 - [Track B: Final Burned Area Prediction](#track-b-final-burned-area-prediction)
- [Contributing](#contributing)
- [Datacube Details](#datacube-details)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Data repository

You can access the data using this [Drive link](https://drive.google.com/drive/folders/1aRXQXVvw6hz0eYgtJDoixjPQO-_bRKz9). This link contains the mesogeos datacube (`mesogeos_cube.zarr/`), the extracted datasets for the machine learning tracks (`ml_tracks/`), as well as notebooks showing how to access the mesogeos cubes (`notebooks/`).

### Accessing the mesogeos cube

The mesogeos cube is publicly accessible in the following places:

- Google Drive folder: [https://drive.google.com/drive/folders/1aRXQXVvw6hz0eYgtJDoixjPQO-_bRK z9
](https://drive.google.com/drive/folders/1aRXQXVvw6hz0eYgtJDoixjPQO-_bRKz9)

#### Option 1: Access from S3 (Best option to download)

```
import zarr
import xarray as xr
import fsspec

url = 'https://my-uc3-bucket.s3.gra.io.cloud.ovh.net/mesogeos.zarr'
ds = xr.open_zarr(fsspec.get_mapper(url))
ds
```

To run this make sure to install `xarray`, `zarr` and `fsspec` libraries. 

**Downloading locally:** You can write the zarr using the [xarray `.to_zarr` method](https://docs.xarray.dev/en/latest/generated/xarray.Dataset.to_zarr.html).

#### Option 2: Access from Google Colab
[notebooks/1_Exploring_Mesogeos.ipynb](notebooks/1_Exploring_Mesogeos.ipynb) shows how to open Mesogeos directly in google colab 
[![colab_link](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Orion-AI-Lab/mesogeos/blob/main/notebooks/1_Exploring_Mesogeos.ipynb)

## Datacube Generation

Find the code to generate a datacube like mesogeos in [datacube_creation](datacube_creation/).

## Machine Learning Tracks
### Track A: Wildfire Danger Forecasting

This track defines wildfire danger forecasting as a binary classification problem.

More details in [Track A](./ml_tracks/a.fire_danger/)

### Track B: Final Burned Area Prediction

This track is about predicting the final burned area of a wildfire given the ignition point and the conditions of the fire drivers at the first day of the fire in a neighborhood around the ignition point.

More details in [Track B](./ml_tracks/b.final_burned_area/README.md)

## Datacube Details

Mesogeos is meant to be used to develop models for wildfire modeling in the Mediterranean. 
It contains variables related to the ignition and spread of wildfire for the years 2006 to 2022 at a daily 1km x 1km grid.

<details> <summary>Datacube Variables</summary>

The datacube contains the following variables:

- satellite data from MODIS (Land Surface Temperature (https://lpdaac.usgs.gov/products/mod11a1v061/), Normalized Vegetation Index (https://lpdaac.usgs.gov/products/mod13a2v061/), Leaf Area Index (https://lpdaac.usgs.gov/products/mod15a2hv061/))
- weather variables from ERA5-Land (max daily temperature, max daily dewpoint temperature, min daily relative humidity, 
max daily wind speed, max daily surface pressure, mean daily surface solar radiation downwards) (https://cds.climate.copernicus.eu/cdsapp#!/dataset/10.24381/cds.e2161bac?tab=overview)
- soil moisture index from JRC European Drought Observatory (https://edo.jrc.ec.europa.eu/edov2/home.static.html)
- population count (https://hub.worldpop.org/geodata/listing?id=64) & distance to roads (https://hub.worldpop.org/geodata/listing?id=33) from worldpop.org 
- land cover from Copernicus Climate Change Service (https://cds.climate.copernicus.eu/cdsapp#!/dataset/satellite-land-cover?tab=overview)
- elevation, aspect, slope and curvature from Copernicus EU-DEM (https://land.copernicus.eu/imagery-in-situ/eu-dem/eu-dem-v1.1?tab=download)
- burned areas and ignition points from EFFIS (https://effis.jrc.ec.europa.eu/applications/data-and-services)

Vriables in the cube:
| Variable | Units | Description |
| --- | --- | --- |
| aspect | ° | aspect |
| burned areas | unitless | rasterized burned polygons. 0 when no burned area occurs in that cell, 1 if it does for the day of interest |
| curvature | rad | curvature |
| d2m | K | day's maximum 2 metres dewpoint temperature |
| dem | m | elevation |
| ignition_points | hectares | rasterized fire ignitions. It contains the final hectares of the burned area resulted from the fire |
| lai | unitless | leaf area index |
| lc_agriculture | % | fraction of agriculture in the pixel. 1st Jan of each year has the values of the year |
| lc_forest | % | fraction of forest in the pixel. 1st Jan of each year has the values of the year |
| lc_grassland | % | fraction of grassland in the pixel. 1st Jan of each year has the values of the year |
| lc_settlement | % | fraction of settlement in the pixel. 1st Jan of each year has the values of the year |
| lc_shrubland | % | fraction of shrubland in the pixel. 1st Jan of each year has the values of the year |
| lc_sparse_veagetation | % | fraction of sparse vegetation in the pixel. 1st Jan of each year has the values of the year |
| lc_water_bodies | % | fraction of water bodies in the pixel. 1st Jan of each year has the values of the year |
| lc_wetland | % | fraction of wetland in the pixel. 1st Jan of each year has the values of the year |
| lst_day | K | day's land surface temperature |
| lst_night | K | nights' land surface temperature |
| ndvi | unitless | normalized difference vegetation index |
| population | people/km^2 | population count per year. 1st Jan of each year has the values of the year |
| rh | %/100 | day's minimum relative humidity |
| roads_distance | km | distance from the nearest road |
| slope | rad | slope |
| smi | unitless | soil moisture index |
| sp | Pa | day's maximum surface pressure |
| ssrd | J/m^2| day's average surface solar radiation downwards |
| t2m | K | day's maximum 2 metres temperature |
| tp | m | day's total precipitation |
| wind_speed | m/s | day's maximum wind speed |

</details>

An example of some variables for a day in the cube:
![image](https://user-images.githubusercontent.com/76213770/225653285-754a7d4a-8f32-4200-820b-d3614e14b864.png)


**Datacube Metadata**

- Temporal Extent: `(2006-04-01, 2022-09-29)`
- Spatial Extent: `(-10.72, 30.07, 36.74, 47.7)`, i.e. the wider Mediterranean region.
- Coordinate Reference System: `EPSG:4326`


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7741518.svg)](https://doi.org/10.5281/zenodo.7741518)

**Datacube Citation** 

```
Spyros Kondylatos, Ioannis Prapas, Gustau Camps-Valls, & Ioannis Papoutsis. (2023). 
Mesogeos: A multi-purpose dataset for data-driven wildfire modeling in the Mediterranean. 
Zenodo. https://doi.org/10.5281/zenodo.7473331
```

## Contributing

We welcome new contributions for new models and new machine learning tracks!

**New Model**: To contribute a new model for an existing track, your code has to be (i) open, (ii) reproducible (we should be able to easily run your code and get the reported results) and (iii) use the same dataset split defined for the track. 
After we verify your results, you get to **add your model and name to the leaderboard**. 
Check the current [leaderboards](https://orion-ai-lab.github.io/mesogeos/).

[Submit a new issue](https://github.com/Orion-AI-Lab/mesogeos/issues/new/choose) containing a link to your code.

**New ML Track**: To contribute a new track, [submit a new issue](https://github.com/Orion-AI-Lab/mesogeos/issues/new/choose).

We recommend at minimum:

1. a dataset extraction process that samples from mesogeos,
2. a description of the task,
3. a baseline model,
4. appropriate metrics.

### License

Creative Commons Attribution v4

### Citation

```
@inproceedings{
kondylatos2023mesogeos,
title={Mesogeos: A multi-purpose dataset for data-driven wildfire modeling in the Mediterranean},
author={Spyros Kondylatos and Ioannis Prapas and Gustau Camps-Valls and Ioannis Papoutsis},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2023},
url={https://openreview.net/forum?id=VH1vxapUTs}
}
```

### Acknowledgements 

This work has received funding from the European Union’s Horizon 2020 Research and Innovation Projects DeepCube and TREEADS, under Grant Agreement Numbers 101004188 and 101036926353 respectively
