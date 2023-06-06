from src import utils
from src.cube import Cuber
from config import logger

import argparse
from pathlib import Path
from joblib import Parallel, delayed

import geopandas as gpd
import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--modis_data_path', type=str, default='jh-shared/iprapas/uc3',
                    help='Path for the data. Default "jh-shared/iprapas/uc3"')
parser.add_argument('--era5_data_path', type=str, default='jh-shared/iprapas/uc3',
                    help='Path for the data. Default "jh-shared/iprapas/uc3"')
parser.add_argument('--ref_data_path', type=str, default='jh-shared/iprapas/uc3',
                    help='Path for the data. Default "jh-shared/iprapas/uc3"')
parser.add_argument('--results_path', type=str, default='jh-shared/med/',
                    help='The path to save results into. Default "js-shared/iprapas/uc3"')
parser.add_argument('start_date', type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d').date(), default='2002-04-01',
                    help='The start date of the cube in format "Year-Month-Day"')
parser.add_argument('end_date', type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d').date(), default='2022-10-01',
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

    results_path = Path.home() / args.results_path

    ref_path = Path.home() / args.ref_data_path

    aoi = gpd.read_file(args.AoI)
    aoi_total_bounds = aoi.total_bounds

    variables = {}

    variables['MOD11A1.006'] = ['LST_Day_1km', 'LST_Night_1km']
    variables['MOD13A2.061'] = ['1 km 16 days NDVI']
    variables['MOD15A2H.061'] = ['Fpar_500m', 'Lai_500m']

    timestep_vars = [
        'ndvi',
        'lst_day',
        'lst_night',
        'lai'
    ]

    static_vars = []

    datacube = Cuber(ds_path=results_path, timestep_vars=timestep_vars, static_vars=static_vars, epsg_crs=args.epsg,
                     region=aoi_total_bounds)

    datacube.init_from_datacube(ref_path, args.start_date, args.end_date)

    files_dict = {}
    for product in ['MOD11A1.006', 'MOD15A2H.061', 'MOD13A2.061']:
        files_dict[product] = utils.create_modis_dicts(modis_data_path / product, args.modis_tiles)
        results = Parallel(n_jobs=-1)(
            delayed(datacube.write_dynamic_var)(product, files_dict[product][f], variables[product]) for f in
            files_dict[product])
        logger.info(results)


if __name__ == "__main__":
    main()