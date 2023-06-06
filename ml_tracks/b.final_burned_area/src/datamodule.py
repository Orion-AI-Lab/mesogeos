from typing import Optional, Tuple
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import numpy as np
import xarray as xr
from collections import defaultdict
import pickle
from pathlib import Path
import random


class BatcherDS(Dataset):
    """Dataset from Xbatcher"""

    def __init__(self, samples, input_vars, target, mean_std_dict=None, min_max_dict=None,  crop_size=0):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.samples = samples
        self.target = target
        self.input_vars = input_vars
        self.mean_std_dict = mean_std_dict
        self.min_max_dict = min_max_dict
        if mean_std_dict:
            self.mean = np.stack([mean_std_dict[f'{var}_mean'] for var in input_vars])
            self.std = np.stack([mean_std_dict[f'{var}_std'] for var in input_vars])
        if min_max_dict:
            self.min = np.stack([min_max_dict[var]['min'] for var in input_vars])
            self.max = np.stack([min_max_dict[var]['max'] for var in input_vars])
        
        self.crop_size = crop_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx].isel(time=-1)
        # make var ignition points binary
        sample['ignition_points'] = sample['ignition_points'].where(sample['ignition_points'] == 0, 1)

        inputs = np.stack([sample[var] for var in self.input_vars]).astype(np.float32)
        for i, var in enumerate(self.input_vars):
            if not var.startswith('lc_') and (var != 'ignition_points'):
                if self.mean_std_dict:
                    inputs[i] = (inputs[i] - self.mean[i]) / self.std[i]
                elif self.min_max_dict:
                    inputs[i] = (inputs[i] - self.min[i]) / (self.max[i] - self.min[i])
                # inputs[i] = (inputs[i] - self.mean_std_dict[f'{var}_mean']) / self.mean_std_dict[f'{var}_std']
        target = sample[self.target].values

        # impute nans in input with the mean
        #  


        inputs = np.nan_to_num(inputs, nan=0.0)
        target = np.nan_to_num(target, nan=0.0)
        # make this a classification dataset
        target = np.where(target != 0, 1, 0)
        # random crop a crop_size x crop_size patch
        len_x = inputs.shape[1]
        len_y = inputs.shape[2]

        if self.crop_size > 0:
            start_x = random.randint(0, len_x - self.crop_size)
            start_y = random.randint(0, len_y - self.crop_size)
            end_x = start_x + self.crop_size
            end_y = start_y + self.crop_size
            inputs = inputs[:, start_x:end_x, start_y:end_y]
            target = target[start_x:end_x, start_y:end_y]
        return inputs, target
    
# function to get sample paths, using dataset_dir and year
def get_sample_paths(dataset_dir, year):
    return list((dataset_dir / f'{year}').glob('*.nc'))

from tqdm import tqdm

class TheDataModule(LightningDataModule):
    """LightningDataModule.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
            self,
            dataset_path: str,
            input_vars : list,
            target: str,
            batch_size: int = 64,
            num_workers: int = 8,
            pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.ds = None
        self.batches = None
        # read mean_std_dict from dataset_path / mean_std_dict.pkl
        with open(f'{dataset_path}/mean_std_dict.pkl', 'rb') as f:
            self.mean_std_dict = pickle.load(f)
        with open(f'{dataset_path}/min_max_dict.pkl', 'rb') as f:
            self.min_max_dict = pickle.load(f)
        self.input_vars = input_vars
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.target = target

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.dataset_path = Path(dataset_path)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            train_years = list(range(2006, 2020))
            val_years = list(range(2020, 2022))
            test_years = [2022]

            # get train sample paths
            train_sample_paths = []
            for year in tqdm(train_years):
                train_sample_paths += get_sample_paths(self.dataset_path, year)

            # get val sample paths
            val_sample_paths = []
            for year in tqdm(val_years):
                val_sample_paths += get_sample_paths(self.dataset_path, year)

            # get test sample paths
            test_sample_paths = []
            for year in tqdm(test_years):
                test_sample_paths += get_sample_paths(self.dataset_path, year)        
            
            def _open_samples(path_list):
                return [xr.open_dataset(path) for path in path_list]

            print('Loading training data...')
            self.data_train = BatcherDS(_open_samples(train_sample_paths), self.input_vars, self.target, min_max_dict = self.min_max_dict, crop_size=32)
            print('Loading validation data...')
            self.data_val = BatcherDS(_open_samples(val_sample_paths),self.input_vars, self.target, min_max_dict = self.min_max_dict, crop_size=0)
            print('Loading test data...')
            self.data_test = BatcherDS(_open_samples(test_sample_paths), self.input_vars, self.target, min_max_dict = self.min_max_dict, crop_size=0)

    def train_dataloader(self):
        return MultiEpochsDataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            persistent_workers=(self.num_workers > 0),
            # drop_last=True
        )

    def val_dataloader(self):
        return MultiEpochsDataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=(self.num_workers > 0),
            # drop_last=True
        )

    def test_dataloader(self):
        return MultiEpochsDataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=(self.num_workers > 0),
            # drop_last=True
        )


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
