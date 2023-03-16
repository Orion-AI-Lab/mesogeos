# mesogeos

mesogeos is meant to be used to develop models for next-day fire hazard forecasting in the Mediterranean. 
It contains variables related to the ignition and spread of wildfire for the years 2002 to 2022 at a daily 1km x 1km grid.

Particularly, it contains :
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
| curvature | | curvature |
| d2m | K | day's maximum 2 metres dewpoint temperature |
| dem | m | elevation |
| ignition_points | unitless | rasterized fire ignitions. 0 when no fire ignition occurs in that cell, 1 if it does for the day of interest |
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
| population | humans/km^2 | population count per year. 1st Jan of each year has the values of the year |
| rh | %/100 | day's minimum relative humidity |
| roads_distance | km | distance from the nearest road |
| slope | | slope |
| smi | unitless | soil moisture index |
| sp | Pa | day's maximum surface pressure |
| ssrd | | day's average surface solar radiation downwards |
| t2m | K | day's maximum 2 metres temperature |
| tp | m | day's total precipitation |
| wind_speed | m/s | day's maximum wind speed |

An example of some variables for a day in the cube:
![image](https://user-images.githubusercontent.com/76213770/225653285-754a7d4a-8f32-4200-820b-d3614e14b864.png)


The full datacube can be downloaded from here: https://drive.google.com/drive/folders/1P_KLpmslD2wePHxaAtHWVduaNatDg74e?usp=share_link

Details of the data cube:
- temporal_extent : (2002-04-01, 2022-09-30)
- spatial_extent : (-10.72, 30.07, 36.74, 47.7)
- crs : EPSG:4326

Creators: Spyros Kondylatos, Ioannis Prapas, Ioannis Papoutsis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7741518.svg)](https://doi.org/10.5281/zenodo.7741518)

Cite as: Spyros Kondylatos, Ioannis Prapas, & Ioannis Papoutsis. (2023). mesogeos: A Daily Datacube for the Modeling and Analysis of Wildfires in the Mediterranean (v1.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7741518

License : Creative Commons Attribution v4

Acknowledgements : This work has received funding from the European Union’s Horizon 2020 research and innovation project DeepCube, under grant agreement number 101004188.
