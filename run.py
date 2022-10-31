from src import utils
from src.cube import Cuber
# from config import logger

import argparse
from pathlib import Path
from joblib import Parallel, delayed

import geopandas as gpd
import datetime
import gc

import os

parser = argparse.ArgumentParser()

parser.add_argument('--modis_data_path', type=str, default='jh-shared/iprapas/uc3/MED/MODIS',
                    help='Path for the data. Default "jh-shared/iprapas/uc3"')
parser.add_argument('--era5_data_path', type=str, default='jh-shared/iprapas/uc3/MED/ERA5-Land',
                    help='Path for the data. Default "jh-shared/iprapas/uc3"')
parser.add_argument('--soil_moisture_data_path', type=str, default='jh-shared/iprapas/uc3/MED/SMI',
                    help='Path for the data. Default "jh-shared/iprapas/uc3"')
parser.add_argument('--burned_areas_data_path', type=str, default='jh-shared/iprapas/uc3/MED/BURNED_AREAS',
                    help='Path for the data. Default "jh-shared/iprapas/uc3"')
parser.add_argument('--dem_data_path', type=str, default='jh-shared/iprapas/uc3/MED/dem_products.nc',
                    help='Path for the data. Default "jh-shared/iprapas/uc3"')
parser.add_argument('--ref_data_path', type=str, default='',
                    help='Path for the data. Default ""')
parser.add_argument('--results_path', type=str, default='jh-shared/med_cube/med_cube_test',
                    help='The path to save results into. Default "jh-shared/iprapas/uc3"')
parser.add_argument('--start_date', type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d').date(), default='2002-04-01',
                    help='The start date of the cube in format "Year-Month-Day"')
parser.add_argument('--end_date', type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d').date(), default='2022-10-01',
                    help='The end date of the cube in format "Year-Month-Day"')
parser.add_argument("--modis_tiles", nargs="+",
                    default=["h17v04", "h18v04", "h19v04", "h20v04", "h17v05", "h18v05", "h19v05", "h20v05"],
                    help='The modis tiles')
parser.add_argument('--epsg', type=int, default=4326,
                    help='The epsg for reprojecting the dataset')
parser.add_argument('--AoI', type=str, default='greece_big.geojson',
                    help='Geojson with area of interest')

args = parser.parse_args()


def main():
    modis_data_path = Path.home() / args.modis_data_path
    era5_data_path = Path.home() / args.era5_data_path
    sm_data_path = Path.home() / args.soil_moisture_data_path
    bas_data_path = Path.home() / args.burned_areas_data_path
    dem_data_path = Path.home() / args.dem_data_path

    results_path = Path.home() / args.results_path

    ref_path = Path.home() / args.ref_data_path

    aoi = gpd.read_file(Path.home() / args.AoI)
    aoi_total_bounds = aoi.total_bounds

    variables_modis = {}

    variables_modis['MOD11A1.061'] = ['LST_Day_1km', 'LST_Night_1km']
    variables_modis['MOD13A2.061'] = ['1 km 16 days NDVI']
    variables_modis['MOD15A2H.061'] = ['Lai_500m']

    timestep_vars = [
        'ndvi',
        'lst_day',
        'lst_night',
        'lai',
        't2m',
        'wind_speed',
        'rh',
        'd2m',
        'sp',
        'ssrd',
        'tp',
        'smi',
        'burned_areas',
        'ignition_points'
    ]

    static_vars = ['dem',
                   'slope',
                   'aspect',
                   'curvature']

    datacube = Cuber(ds_path=results_path, timestep_vars=timestep_vars, static_vars=static_vars, epsg_crs=args.epsg,
                     region=aoi_total_bounds)

    datacube.init_from_datacube(ref_path, args.start_date, args.end_date)

    #ADD DEM DATA
    datacube.write_static_var('DEM', dem_data_path, ['dem', 'dem_aspect', 'dem_slope_radians', 'dem_curvature'])
    print('DEM products appended to the datacube')

    #ADD BURNED AREAS AND IGNITIONS
    for product in ['BURNED_AREAS', 'IGNITION_POINTS']:
        bas_ds, dates_with_fire = utils.read_burned_areas(bas_data_path, product)
        _ = Parallel(n_jobs=16)(
                delayed(datacube.write_dynamic_var)(product, utils.create_bas_files(bas_ds, dates_with_fire, dt, product),
                                                    [product.lower()])
                for dt in
                [args.start_date + datetime.timedelta(days=i) for i in range((args.end_date-args.start_date).days + 1)])
        print(product + ' appended to the cube')

    # ADD MODIS
    files_dict = {}
    for product in ['MOD11A1.061', 'MOD15A2H.061', 'MOD13A2.061']:
        files_dict[product] = utils.create_modis_dicts(modis_data_path / product, args.modis_tiles,
                                                       args.start_date, args.end_date)
        _ = Parallel(n_jobs=16)(
            delayed(datacube.write_dynamic_var)(product, files_dict[product][f], variables_modis[product]) for f in
            files_dict[product])
        print(product + ' appended to the cube')


    # ADD SOIL MOISTURE
    filenames = os.listdir(sm_data_path)
    for year in range(args.start_date.year, args.end_date.year + 1):
        sm_ds = utils.open_smi_ds(sm_data_path, filenames, year)
        _ = Parallel(n_jobs=16)(
            delayed(datacube.write_dynamic_var)('SMI', utils.create_sm_files(sm_ds, i),
                                                ['sminx'])
            for i in range(len(sm_ds['time'])))
    print('Soil Moisture appended to the cube')


    #ADD ERA5_LAND
    filenames = os.listdir(era5_data_path)
    for year in range(args.start_date.year, args.end_date.year + 1):
        era5_ds = utils.open_era5_ds(era5_data_path, filenames, year)
        _ = Parallel(n_jobs=16)(
                delayed(datacube.write_dynamic_var)('ERA5-Land', utils.create_era5_files(era5_ds, i),
                ['era5_max_t2m', 'era5_max_wind_speed', 'era5_min_rh', 'era5_max_d2m', 'era5_max_sp', 'era5_max_ssrd', 'era5_max_tp'])
                for i in range(int(len(era5_ds['time'])/24)))
    print('ERA5-Land appended to the cube')


if __name__ == "__main__":
    main()
