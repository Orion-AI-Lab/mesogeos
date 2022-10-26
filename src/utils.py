import os
import xarray as xr
import datetime


def create_modis_dicts(input_path, modis_tiles, start_date, end_date):
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
    return files_dict

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