# mesogeos

mesogeos is meant to be used to develop models for next-day fire hazard forecasting in the Mediterranean. 
It contains variables related to the ignition and spread of wildfire for the years 2002 to 2022 at a daily 1km x 1km grid.
Particularly, it contains satellite data from MODIS (Land Surface Temperature, Normalized Vegetation Index, Leaf Area Index), 
weather variables from ERA5-Land (max daily temperature, max daily dewpoint temperature, min daily relative humidity, 
max daily wind speed, max daily surface pressure, mean daily surface solar radiation downwards), soil moisture index from
JRC European Drought Observatory, population count & distance to roads from worldpop.org, land cover from Copernicus 
Climate Change Service and elevation, aspect, slope and curvature from Copernicus EU-DEM and burned areas and ignition 
points from EFFIS.

The full datacube can be downloaded from here: 

Details of the data cube:

temporal_extent : (2002-04-01, 2022-09-30)
spatial_extent : (-10.72, 30.07, 36.74, 47.7)
crs : EPSG:4326