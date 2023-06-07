import os
import xarray as xr
import datetime
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import wkt


def create_modis_dicts(input_path, modis_tiles, start_date, end_date, product):
    files = os.listdir(input_path)
    files.sort()

    files_dict = {}
    for f in files:
        if (f.endswith('hdf')) and (f.split('.')[2] in modis_tiles):
            date = f.split('.')[1][1:]
            date = datetime.datetime.strptime(date, '%Y%j').date()
            if (date >= start_date) and (date <= end_date):
                if date in files_dict:
                    files_dict[date].append(input_path / f)
                else:
                    files_dict[date] = [input_path / f]
    if product == 'MOD11A1.061':
        # del files_dict[datetime.datetime.strptime('2003358', '%Y%j').date()]
        del files_dict[datetime.datetime.strptime('2010064', '%Y%j').date()]
        files_dict[datetime.datetime.strptime('2006172', '%Y%j').date()] = files_dict[
            datetime.datetime.strptime('2006171', '%Y%j').date()]
        files_dict[datetime.datetime.strptime('2015190', '%Y%j').date()] = files_dict[
            datetime.datetime.strptime('2015189', '%Y%j').date()]
        files_dict[datetime.datetime.strptime('2021265', '%Y%j').date()] = files_dict[
            datetime.datetime.strptime('2021264', '%Y%j').date()]
        files_dict[datetime.datetime.strptime('2019344', '%Y%j').date()] = files_dict[
            datetime.datetime.strptime('2019343', '%Y%j').date()]
        files_dict[datetime.datetime.strptime('2017085', '%Y%j').date()] = files_dict[
            datetime.datetime.strptime('2017084', '%Y%j').date()]
        files_dict[datetime.datetime.strptime('2019169', '%Y%j').date()] = files_dict[
            datetime.datetime.strptime('2019168', '%Y%j').date()]
    if product == 'MOD15A2H.061':
        files_dict[datetime.datetime.strptime('2019233', '%Y%j').date()] = files_dict[
            datetime.datetime.strptime('2019225', '%Y%j').date()]

    return files_dict


def read_dem(dataset_path):
    return xr.open_dataset(dataset_path)


def read_burned_areas(dataset_path, product):
    common_crs = 'epsg:4326'

    ba_df = gpd.read_file(dataset_path)
    ba_df['IGNITION_DATE'] = pd.to_datetime(ba_df.IGNITION_D).dt.date
    if product == 'IGNITION_POINTS':
        ba_df['geometry_h'] = ba_df['geometry_h'].apply(wkt.loads)
        ba_df = ba_df.set_geometry('geometry_h')
    ba_df.crs = "EPSG:5643"
    ba_df = ba_df.to_crs(common_crs)
    list_of_dates = ba_df['IGNITION_DATE'].tolist()

    return ba_df, list_of_dates


def create_bas_files(bas_ds, list_of_dates, i, product):
    if i + pd.Timedelta("1 days") in list_of_dates:
        ind = bas_ds.index[bas_ds['IGNITION_DATE'] == i + pd.Timedelta("1 days")]
        if product == 'IGNITION_POINTS':
            return [bas_ds.iloc[ind].geometry_h.tolist(), bas_ds.iloc[ind].AREA_HA.tolist(), np.datetime64(i)]
        elif product == 'BURNED_AREAS':
            return [bas_ds.iloc[ind].geometry.tolist(), [1]*len(bas_ds.iloc[ind].geometry.tolist()), np.datetime64(i)]
    else:
        return [[], [], np.datetime64(i)]


def create_era5_files(era5_ds, i):
    ds_temp = era5_ds.isel(time=slice(i*24, (i+1)*24))
    return ds_temp


def create_sm_files(sm_ds, i):
    ds_temp = sm_ds.isel(time=slice(i, i+1))
    return ds_temp


def open_era5_ds(data_path, filenames, year):
    file = [f for f in filenames if str(year) in f][0]
    path = data_path / file
    ds = xr.open_dataset(path)
    return ds


def open_lc_ds(data_path, filenames, year):
    if year <= 2020:
        file = [f for f in filenames if str(year) in f]
    else:
        file = [f for f in filenames if '2020' in f]
    file = file[0]
    path = data_path / file
    ds = xr.open_dataset(path)
    if year != 2006:
        dt = '01-01-' + str(year)
    else:
        dt = '04-01-' + str(year)
    dt = np.datetime64(pd.to_datetime(dt))
    ds['time'].values[0] = dt
    return ds


def open_pop_den_ds(data_path, filenames, year):
    if year <= 2020:
        file = [f for f in filenames if str(year) in f]
    else:
        file = [f for f in filenames if '2020' in f]
    file = file[0]
    path = data_path / file
    ds = xr.open_dataset(path)
    if year != 2006:
        dt = '01-01-' + str(year)
    else:
        dt = '04-01-' + str(year)
    pop_den_year = np.datetime64(pd.to_datetime(dt))
    return ds, pop_den_year


def open_smi_ds(data_path, filenames, year):
    file = [f for f in filenames if str(year) in f and 'sminx' in f][0]
    path = data_path / file
    ds = xr.open_dataset(path)
    return ds


def str2bool(v):
    valid = {'True': True, 'true': True, 't': True, '1': True,
             'False': False, 'false': False, 'f': False, '0': False,
             }
    lower_value = v.lower()
    if lower_value in valid:
        return valid[lower_value]
    else:
        raise ValueError('invalid literal for boolean: "%s"' % v)