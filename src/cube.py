import xarray as xr
from collections import Counter
import pathlib


# list files in a pathlib.Path directory group by the first characters of each filename and count the number of files in each group
def list_files_by_prefix(path, prefix_length):
    files = path.glob("*")
    files = [str(f) for f in files]
    files = [f.split("\\")[-1] for f in files]
    files = [f[:prefix_length] for f in files]
    files = Counter(files)
    return files


class Cuber(object):
    def __init__(self, ds_path=None, chunks=None, reference=None):
        self.ds_path = ds_path
        self.datacube = None

    def init_from_datacube(self, ds_path):
        self.datacube = xr.open_zarr(ds_path)
        self.dims = self.datacube.dims


    def write_dynamic_var(self, region, var_name, var_data):
        pass

    def _write_ndvi(self):
        pass

    def write_spatial_var(self, var_name):
        pass


if __name__ == "__main__":
    # parallel processing
    pass