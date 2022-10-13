from pymodis import downmodis
import cdsapi
import config
import os
import ssl
from config import logger


# define your ~/.netrc file
#
# `
# machine urs.earthdata.nasa.gov
# login YOURUSER
# password YOURPASSWD
# `
#
# and set the correct permissions
#
# `
# chmod og-rw ~/.netrc
# `


def download_modis(output_folder=config.DATASET_PATH, tiles=config.MODIS_SICILY_TILES,
                   fromdate=config.DOWNLOAD_FROM_DATE, todate=config.DOWNLOAD_TO_DATE,
                   products=config.MODIS_PRODUCTS):
    # import os
    # if not os.path.exists(os.path.join(os.path.expanduser('~'), '.netrc')) and not all(user, password)
    for product in products:
        logger.info("Downloading {}...".format(product))
        product_folder = os.path.join(output_folder, product)
        if not os.path.exists(product_folder):
            os.mkdir(product_folder)
        modis_downloader = downmodis.downModis(user=os.environ.get('MODIS_USER'), password=os.environ.get('MODIS_PASS'),
                                               destinationFolder=product_folder,
                                               # url=url,
                                               tiles=tiles,
                                               # path=remote_path,
                                               product=product, today=todate, enddate=fromdate,
                                               debug=True, timeout=60)
        modis_downloader.connect()
        modis_downloader.downloadsAllDay()


def download_era5(fromyear=2009, toyear=2012):
    c = cdsapi.Client()
    era5_dataset_path = config.DATASET_PATH / 'era5'
    if not os.path.exists(era5_dataset_path):
        os.mkdir(era5_dataset_path)
    c.retrieve(
        'reanalysis-era5-land',
        {
            'variable': [
                '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
                '2m_temperature', 'total_precipitation',
            ],
            'year': list(range(fromyear, toyear + 1)),
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': '12:00',
            'area': [
                38.3, 12.55, 36.65,
                15.56,
            ],
            'format': 'netcdf',
        },
        era5_dataset_path / 'era5.nc')


if __name__ == '__main__':
    download_modis(output_folder='./output', fromdate='2020.10.01', todate='2020.10.15')
    download_era5()
