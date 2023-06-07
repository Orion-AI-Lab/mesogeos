import xarray as xr
import numpy as np
import rioxarray as rxr
import datetime
import dask
import pandas as pd
from shapely.geometry import box
import rasterio
from affine import Affine
from rasterio import features
import calendar


class Cuber(object):
    def __init__(self, ds_path=None, timestep_vars=None, static_vars=None, epsg_crs=None,
                 region=None):
        self.ds_path = ds_path
        self.datacube = None
        self.timestep_vars = timestep_vars
        self.static_vars = static_vars
        self.epsg_crs = epsg_crs
        self.region = region
        self.names = self.get_varname_map()
        self.bounds = region

    def get_varname_map(self):
        return {
            'LST_Day_1km': 'lst_day',
            'LST_Night_1km': 'lst_night',
            '1 km 16 days NDVI': 'ndvi',
            'Lai_500m': 'lai',
            'era5_max_t2m': 't2m',
            'era5_max_wind_speed': 'wind_speed',
            'era5_max_wind_direction': 'wind_direction',
            'era5_min_rh': 'rh',
            'era5_max_d2m': 'd2m',
            'era5_max_sp': 'sp',
            'era5_avg_ssrd': 'ssrd',
            'era5_max_tp': 'tp',
            'sminx': 'smi',
            'burned_areas': 'burned_areas',
            'ignition_points': 'ignition_points',
            'dem': 'dem',
            'dem_aspect': 'aspect',
            'dem_slope_radians': 'slope',
            'dem_curvature': 'curvature',
            'population_density': 'population',
            'lc_agriculture': 'lc_agriculture',
            'lc_forest': 'lc_forest',
            'lc_grassland': 'lc_grassland',
            'lc_wetland': 'lc_wetland',
            'lc_settlement': 'lc_settlement',
            'lc_shrubland': 'lc_shrubland',
            'lc_sparse_vegetation': 'lc_sparse_vegetation',
            'lc_water_bodies': 'lc_water_bodies',
            'roads_distance': 'roads_distance'
        }

    def init_from_datacube(self, reference_path, start_date, end_date):
        dc_reference = xr.open_zarr(reference_path)
        x_ = dc_reference['x']
        y_ = dc_reference['y']

        timesteps = [np.datetime64(start_date + datetime.timedelta(days=x)) for x in
                     range((end_date - start_date).days + 1)]

        static_zeros = dask.array.zeros([len(y_), len(x_)], chunks={0: len(x_), 1: len(y_)}, dtype=np.float32)
        timesteps_zeros = dask.array.zeros([len(timesteps)] + [len(y_), len(x_)],
                                           chunks={0: 1, 1: len(x_), 2: len(y_)}, dtype=np.float32)

        data_vars = {}

        data_vars.update({varname: (['y', 'x'], static_zeros) for varname in self.static_vars})
        data_vars.update({varname: (['time', 'y', 'x'], timesteps_zeros) for varname in self.timestep_vars})

        self.datacube = xr.Dataset(data_vars=data_vars,
                                   coords={
                                       'y': y_,
                                       'x': x_,
                                       'time': timesteps,
                                   }
                                   ).chunk({'x': len(x_), 'y': len(y_), 'time': 1})

        self.datacube = self.datacube.rio.write_crs(self.epsg_crs, inplace=True)

        self.datacube.to_zarr(self.ds_path, consolidated=True, compute=False, mode="w")

        return

    def append_to_datacube(self, var_name):
        pass

    def write_static_var(self, product, filename, var_names):

        if product == 'DEM':
            ds = self._dem(filename)
        elif product == 'DST_ROADS':
            ds = self._roads_dst(filename)

        data_vars = {
            self.names[var_name]: (('y', 'x'), ds[var_name].values) for var_name in var_names
        }

        new_data = xr.Dataset(
            data_vars=data_vars,
            coords={
                "x": ds.x,
                "y": ds.y
            },
        ).chunk({'x': self.datacube.chunks['x'][0],
                 'y': self.datacube.chunks['y'][0]})

        if 'spatial_ref' in new_data.variables:
            new_data = new_data.drop(["spatial_ref"])

        if 'band' in new_data.variables:
            new_data = new_data.drop(["band"])

        new_data.to_zarr(self.ds_path,
                         region={
                             'y': slice(0, self.datacube.chunks['y'][0]),
                             'x': slice(0, self.datacube.chunks['x'][0])
                         }
                         )
        return

    def write_dynamic_var(self, product, files, var_names):

        if product in ['MOD11A1.061', 'MOD15A2H.061', 'MOD13A2.061']:
            ds, temporal_resolution = self._modis(files, product)
        elif product == 'ERA5-Land':
            ds, temporal_resolution = self._era5(files)
        elif product == 'SMI':
            ds, temporal_resolution = self._smi(files)
        elif product in ['BURNED_AREAS', 'IGNITION_POINTS']:
            ds, temporal_resolution = self._burned_areas(files, var_names)
        elif product == 'POP_DEN':
            ds, temporal_resolution = self._pop_den(files, var_names)
        elif product == 'LAND_COVER':
            ds, temporal_resolution = self._lc(files, var_names)

        for i in range(temporal_resolution):

            lag = str(i) + ' days'
            timestep = ds.time + pd.Timedelta(lag)

            data_vars = {
                self.names[var_name]: (('time', 'y', 'x'), ds[var_name].values) for var_name in var_names
            }

            new_data = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "x": ds.x,
                    "y": ds.y,
                    "time": timestep
                },
            ).chunk({'x': self.datacube.chunks['x'][0],
                     'y': self.datacube.chunks['y'][0],
                     'time': self.datacube.chunks['time'][0]})

            if 'spatial_ref' in new_data.variables:
                new_data = new_data.drop(["spatial_ref"])

            if 'band' in new_data.variables:
                new_data = new_data.drop(["band"])

            try:
                pos = np.argwhere(self.datacube['time'].values == timestep.values)[0][0]
                new_data.to_zarr(self.ds_path,
                                 region={
                                     'time': slice(pos, pos + 1),
                                     'y': slice(0, self.datacube.chunks['y'][0]),
                                     'x': slice(0, self.datacube.chunks['x'][0])
                                 }
                                 )
            except IndexError:
                pass

        return

    def _dem(self, filename):
        return xr.open_dataset(filename)

    def _roads_dst(self, filename):
        rd_ds = xr.open_dataset(filename).squeeze()
        rd_ds = rd_ds.coarsen(x=10, y=10, boundary='pad').min()
        rd_ds = rd_ds.interp(x=self.datacube['x'], y=self.datacube['y'], method='nearest')
        rd_ds = rd_ds.rename({'band_data': 'roads_distance'})
        return rd_ds


    def _lc(self, filename, var_names):
        tmp_res = 1
        a = np.uint8(1)
        b = np.uint8(0)
        # lc_ds = filename.rio.write_crs(4326)
        lc_ds = filename.sel(lon=slice(self.bounds[0] - 0.05, self.bounds[2] + 0.05),
                          lat=slice(self.bounds[3] + 0.05, self.bounds[1] - 0.05))

        for var in var_names:
            if var == 'lc_agriculture':
                Class = xr.where((lc_ds.lccs_class[:, :, :] >= 10) & (lc_ds.lccs_class[:, :, :] <= 40), a, b)
            elif var == 'lc_forest':
                Class = xr.where((lc_ds.lccs_class[:, :, :] >= 50) & (lc_ds.lccs_class[:, :, :] <= 100), a, b)
            elif var == 'lc_grassland':
                Class = xr.where((lc_ds.lccs_class[:, :, :] == 110) | (lc_ds.lccs_class[:, :, :] == 130), a, b)
            elif var == 'lc_wetland':
                Class = xr.where((lc_ds.lccs_class[:, :, :] >= 160) & (lc_ds.lccs_class[:, :, :] <= 180), a, b)
            elif var == 'lc_settlement':
                Class = xr.where((lc_ds.lccs_class[:, :, :] == 190), a, b)
            elif var == 'lc_shrubland':
                Class = xr.where((lc_ds.lccs_class[:, :, :] >= 120) & (lc_ds.lccs_class[:, :, :] <= 122), a, b)
            elif var == 'lc_sparse_vegetation':
                Class = xr.where((lc_ds.lccs_class[:, :, :] >= 140) & (lc_ds.lccs_class[:, :, :] <= 153) | (
                            lc_ds.lccs_class[:, :, :] >= 200) & (lc_ds.lccs_class[:, :, :] <= 202) |
                                 (lc_ds.lccs_class[:, :, :] == 220), a, b)
            elif var == 'lc_water_bodies':
                Class = xr.where(lc_ds.lccs_class[:, :, :] == 210, a, b)

            classes = Class.coarsen(lon=3, lat=3, boundary='pad').sum()
            classes = classes.where(classes < 0, classes / 9)
            classes = classes.to_dataset()
            classes = classes.rename({'lccs_class': var, 'lat': 'y', 'lon': 'x'})
            classes = classes.interp(x=self.datacube['x'], y=self.datacube['y'])

            if var == 'lc_agriculture':
                final_classes = classes
            else:
                final_classes[var] = classes[var]

        return final_classes, tmp_res


    def _pop_den(self, pop_den, name):
        tmp_res = 1
        ds_pop_den, dt = pop_den
        ds_pop_den = ds_pop_den.squeeze()
        ds_pop_den = ds_pop_den.interp(x=self.datacube['x'].values, y=self.datacube['y'].values, method='nearest')
        ds_pop_den = ds_pop_den.rename({'band_data': name[0]})
        ds_pop_den = ds_pop_den.expand_dims({'time': [np.datetime64(pd.to_datetime(dt))]})
        return ds_pop_den, tmp_res

    def transform_from_latlon(self, lat, lon):
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        trans = Affine.translation(lon[0], lat[0])
        scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
        return trans * scale

    def _burned_areas(self, filenames, var):

        tmp_res = 1

        files, sizes, dt = filenames[0], filenames[1], filenames[2]

        a = np.zeros(shape=(len(self.datacube['y']), len(self.datacube['x'])))

        if files:
            for i in range(len(files)):
                mask = rasterio.features.geometry_mask(
                    [files[i]],
                    out_shape=(len(self.datacube['y']), len(self.datacube['x'])),
                    transform=self.transform_from_latlon(self.datacube['y'], self.datacube['x']),
                    all_touched=True,
                    invert=True)
                mask = mask.astype(float)
                mask[mask == 1] = float(sizes[i])
                a += mask

        a = a[np.newaxis, ...]

        mask = xr.DataArray(a, coords={'time': [dt], 'y': self.datacube['y'].values, 'x': self.datacube['x'].values},
                            dims=["time", "y", "x"])

        mask = mask.rename(var[0]).to_dataset()

        return mask, tmp_res

    def _smi(self, ds_smi):

        ds_smi = ds_smi.rename({'lon': 'x', 'lat': 'y'})

        del ds_smi['3035']
        ds_smi.attrs = {}

        ds_smi = ds_smi.rio.write_crs(3035)
        ds_smi = ds_smi.rio.reproject(self.epsg_crs)
        ds_smi = ds_smi.drop(["lambert_azimuthal_equal_area"])

        ds_smi = ds_smi.interp(x=self.datacube['x'].values, y=self.datacube['y'].values, method='nearest')

        for x in ds_smi:
            ds_smi[x].attrs = {}

        tmp_time = pd.to_datetime(ds_smi['time'].values)

        if tmp_time.month in [1, 3, 5, 7, 8, 10, 12] and tmp_time.day == 21:
            tmp_res = 11
        elif tmp_time.month == 2 and tmp_time.day == 21:
            if calendar.isleap(tmp_time.year):
                tmp_res = 9
            else:
                tmp_res = 8
        else:
            tmp_res = 10

        return ds_smi, tmp_res

    def _modis(self, filenames, product):

        if product == 'MOD11A1.061':
            tmp_res = 1
        elif product == 'MOD13A2.061':
            tmp_res = 16
        elif product == 'MOD15A2H.061':
            tmp_res = 8
        else:
            tmp_res = 1

        ds = []

        for i in range(len(filenames)):
            ds.append(rxr.open_rasterio(filenames[i]))
            ds[i] = ds[i].squeeze()
            ds[i].attrs = {}

        # Specific for MED
        ds1 = xr.concat([ds[0], ds[2], ds[4], ds[6]], dim='x')
        ds2 = xr.concat([ds[1], ds[3], ds[5], ds[7]], dim='x')
        ds = xr.concat([ds1, ds2], dim='y')
        file_date = filenames[0].name.split('.')[1][1:]

        file_date = np.array(datetime.datetime.strptime(file_date, '%Y%j').date(), dtype='datetime64')
        ds.coords['time'] = file_date
        ds = ds.expand_dims('time')

        #         if variables:
        #             ds = ds[variables]
        #         else:
        #             pass

        if self.epsg_crs:
            ds = ds.rio.reproject(self.epsg_crs)

        ds = ds.rio.clip([box(*self.bounds)], all_touched=True, from_disk=True)

        ds = ds.astype('float32')

        for x in ds:
            if product == 'MOD15A2H.061':
                ds[x].attrs['_FillValue'] = [249.0, 250.0, 251.0, 252.0, 253.0, 254.0, 255.0]
            a = np.isin(ds[x].values, ds[x].attrs['_FillValue'])
            ds[x].values[a] = float('NaN')
            if product == 'MOD13A2.061':
                ds[x].attrs['scale_factor'] = 0.0001
            ds[x].values = ds[x].values * ds[x].attrs['scale_factor'] + ds[x].attrs['add_offset']
            ds[x].attrs = {}

        if product == 'MOD15A2H.061':
            ds = ds.interp(x=self.datacube['x'], y=self.datacube['y'], method='nearest')

        return ds, tmp_res

    def _era5(self, ds_temp):

        tmp_res = 1
        era5_ds_max = ds_temp.rename({'longitude': 'x', 'latitude': 'y'})
        era5_ds_max['speed'] = np.sqrt(era5_ds_max['u10'] ** 2 + era5_ds_max['v10'] ** 2)
        wind_speed = era5_ds_max.coarsen(time=24).max()

        max_meter = wind_speed['speed'].to_dataset()

        u10 = era5_ds_max['u10'].where(
            np.sqrt(era5_ds_max['u10'] ** 2 + era5_ds_max['v10'] ** 2).values == max_meter['speed'].values).to_dataset()
        v10 = era5_ds_max['v10'].where(
            np.sqrt(era5_ds_max['u10'] ** 2 + era5_ds_max['v10'] ** 2).values == max_meter['speed'].values).to_dataset()

        u10 = u10.coarsen(time=24).max()
        v10 = v10.coarsen(time=24).max()

        wind_speed['v10'].values = v10['v10'].values
        wind_speed['u10'].values = u10['u10'].values
        wind_speed['direction'] = np.mod(180 + (180 / np.pi) * np.arctan2(wind_speed['u10'], wind_speed['v10']), 360)

        wind_speed['time'] = wind_speed['time'] - pd.Timedelta("1 days") - pd.Timedelta("11.5 H")

        wind = wind_speed['speed'].to_dataset().interp(x=self.datacube['x'].values, y=self.datacube['y'].values,
                                       method='linear').rename(
            {'speed': 'era5_max_wind_speed'})

        wind_speed = wind_speed.interp(x=self.datacube['x'].values, y=self.datacube['y'].values,
                                       method='nearest').rename(
            {i: 'era5_max_wind_{}'.format(i) for i in wind_speed.data_vars})

        wind_speed['era5_max_wind_speed'].values = wind['era5_max_wind_speed'].values

        ds_temp = ds_temp.isel(time=slice(0, 24))
        b = 17.625
        ds_temp['rh'] = np.exp(
            243 * b * (ds_temp['d2m'] - ds_temp['t2m']) / ((ds_temp['t2m'] - 273 + 243) * (ds_temp['d2m'] - 273 + 243)))
        #     ds_temp = ds_temp['rh'].to_dataset()

        era5_ds_max = ds_temp.rename({'longitude': 'x', 'latitude': 'y'}).coarsen(time=24).max()
        era5_ds_max['time'] = era5_ds_max['time'] - pd.Timedelta("1 days") - pd.Timedelta("11.5 H")
        era5_ds_min = ds_temp.rename({'longitude': 'x', 'latitude': 'y'}).coarsen(time=24).min()
        era5_ds_min['time'] = era5_ds_min['time'] - pd.Timedelta("1 days") - pd.Timedelta("11.5 H")
        era5_ds_avg = ds_temp.rename({'longitude': 'x', 'latitude': 'y'}).coarsen(time=24).mean()
        era5_ds_avg['time'] = era5_ds_avg['time'] - pd.Timedelta("1 days") - pd.Timedelta("11.5 H")

        era5_ds_max = era5_ds_max.interp(x=self.datacube['x'].values, y=self.datacube['y'].values,
                                         method='linear').rename(
            {i: 'era5_max_{}'.format(i) for i in ds_temp.data_vars})
        era5_ds_min = era5_ds_min.interp(x=self.datacube['x'].values, y=self.datacube['y'].values,
                                         method='linear').rename(
            {i: 'era5_min_{}'.format(i) for i in ds_temp.data_vars})
        era5_ds_avg = era5_ds_avg.interp(x=self.datacube['x'].values, y=self.datacube['y'].values,
                                         method='linear').rename(
            {i: 'era5_avg_{}'.format(i) for i in ds_temp.data_vars})

        ds = era5_ds_max
        for var in era5_ds_min:
            ds[var] = era5_ds_min[var]
        for var in era5_ds_avg:
            ds[var] = era5_ds_avg[var]
        for var in wind_speed:
            ds[var] = wind_speed[var]

        return ds, tmp_res


if __name__ == "__main__":
    # parallel processing
    pass
