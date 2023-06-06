# %%
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from shapely import wkt
from collections import defaultdict
import argparse


def preprocess_ba_gdf(gdf):
    # change the CRS of gdf to EPSG:4326
    # slice gdf to rows with IGNITION_D >= '2006-04-01'
    gdf = gdf[gdf['IGNITION_D'] >= '2006-04-02'].reset_index(drop=True)
    gdf = gdf[gdf['IGNITION_D'] <= '2022-09-30'].reset_index(drop=True)
    gdf = gdf[gdf['AREA_HA'].astype(float) >= 30]
    # convert FIREDATE column to datetime
    gdf['IGNITION_D'] = pd.to_datetime(gdf['IGNITION_D'])
    # subtract 1 day from FIREDATE
    gdf['IGNITION_D'] = gdf['IGNITION_D'] - pd.Timedelta(days=1)
    # get the year of FIREDATE
    gdf['IGNITION_YEAR'] = gdf['IGNITION_D'].dt.year
    # change geometry
    gdf['geometry_h'] = gdf['geometry_h'].apply(wkt.loads)
    gdf = gdf.set_geometry('geometry_h')
    gdf.crs = "EPSG:5643"
    gdf = gdf.to_crs(epsg=4326)
    gdf = gdf.reset_index(drop=True)
    return gdf

def get_static_cache(static_vars : list=None, dataset: xr.Dataset=None):
    """
    Creates a cache for static variables.
    """
    if static_vars is None:
            static_vars = [
                'lc_agriculture',
                'lc_forest',
                'lc_grassland',
                'lc_settlement',
                'lc_shrubland',
                'lc_sparse_vegetation',
                'lc_water_bodies',
                'lc_wetland',
                'population']


    cache_dict = defaultdict(str)
    # get unique years from ds.time
    years = list(np.unique(ds.time.dt.year.values))
    for year in years: 
        for var in static_vars:
            if str(year) == str(2006):
                cache_dict[f'{var}_{year}'] = ds[var].isel(time=0).load()
            else:
                cache_dict[f'{var}_{year}'] = ds[var].sel(time=f'{year}-01-01').load()
    return cache_dict


def extract_dataset(ds, gdf, patch_half, days_before, days_after, cache_dict, max_samples=1000):
    """
    Extracts a dataset from a xarray dataset for a given geodataframe.
    The dataset is extracted for a given shape.
    The dataset is extracted for a given number of days before and after the ignition date.
    """
    samples = []
    len_x = len(ds.x)
    len_y = len(ds.y)
    date_format = '%Y-%m-%d'  
    gdf.reset_index(drop=True, inplace=True)      
    for i in tqdm(range(min(len(gdf), max_samples))):
        try:

            ignition_date = gdf.loc[i, 'IGNITION_D']
            ignition_xy = gdf.loc[i, 'geometry_h']

            year = ignition_date.year
            
            # ign_date_str  = (ignition_date).strftime(date_format)
            before_date_lag_str = (ignition_date - pd.Timedelta(days=days_before-1)).strftime(date_format)
            after_date_lag_str = (ignition_date + pd.Timedelta(days=days_after)).strftime(date_format)


                    
                            
            sample_ds = ds.sel(time=slice(before_date_lag_str, after_date_lag_str))
            
            sample = sample_ds.sel(x=ignition_xy.x, y=ignition_xy.y, method='nearest')
            x_idx = np.where(sample_ds['x']==sample['x'].values)[0].item()
            y_idx = np.where(sample_ds['y']==sample['y'].values)[0].item()

            if ((x_idx - patch_half < 0) or (x_idx + patch_half + 1 >= len_x) or (y_idx - patch_half < 0) or (y_idx + patch_half + 1 >= len_y)):
                print('border')
                continue


            year = str(ignition_date.year)
            sample_ds = sample_ds.isel(x=slice(x_idx - patch_half,x_idx + patch_half),
                                            y=slice(y_idx - patch_half,y_idx + patch_half))
            
            for var in [x for x in sample_ds.data_vars if x == 'population' or 'lc' in x]:
                del sample_ds[var]
                sample_ds[var] = cache_dict[f'{var}_{year}'].sel(x=sample_ds.x, y=sample_ds.y)
            samples.append(sample_ds.load())
        except Exception as e:
            print(e)
            continue
    return samples


def save_dataset(dataset, save_dir):
    """
    Saves a dataset to a given directory.
    """
    save_dir = Path(save_dir)
    for i, sample in tqdm(enumerate(dataset)):
        # get string of year from sample
        year = str(sample.time.dt.year.values[0])
        year_dir = save_dir / year
        year_dir.mkdir(parents=True, exist_ok=True)
        sample.to_netcdf(year_dir /  f'sample_{i}.nc')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_half', type=int, default=32, help='Half of the patch size')
    parser.add_argument('--days_before', type=int, default=1, help='Number of days before the ignition date')
    parser.add_argument('--days_after', type=int, default=5, help='Number of days after the ignition date')
    parser.add_argument('--save_dir', type=str, help='Directory to save the dataset')
    parser.add_argument('--ds_path', type=str, help='Path to the zarr dataset')
    parser.add_argument('--gdf_path', type=str, help='Path to the geodataframe containing polygons of the burned areas')
    args = parser.parse_args()
    ds = xr.open_dataset(args.ds_path)
    gdf = gpd.read_file(args.gdf_path)
    gdf = preprocess_ba_gdf(gdf)
    static_vars = [
        'lc_agriculture',
        'lc_forest',
        'lc_grassland',
        'lc_settlement',
        'lc_shrubland',
        'lc_sparse_vegetation',
        'lc_water_bodies',
        'lc_wetland',
        'population']

    cache_dict = get_static_cache(static_vars=static_vars, dataset=ds)
    dataset = extract_dataset(ds, gdf, args.patch_half, args.days_before, args.days_after, cache_dict, max_samples=100000)
    save_dataset(dataset, args.save_dir)
