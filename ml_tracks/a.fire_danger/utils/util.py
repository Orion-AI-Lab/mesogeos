import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from sklearn.metrics import average_precision_score


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use, num_device):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:' + str(num_device) if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for ind in self._data.index:
            for col in self._data.columns:
                if ind not in ['auc', 'aucpr', 'ece', 'spread_skill']:
                    self._data.loc[ind][col] = 0
                else:
                    self._data.loc[ind][col] = []

    def update(self, key, nominator, denominator):
        if self.writer is not None:
            self.writer.add_scalar(key, nominator, denominator)
        self._data.total[key] += nominator
        self._data.counts[key] += denominator
        self._data.average[key] = self._data.total[key] / self._data.counts[key]


    def aucpr_update(self, key, preds, labels):
        if self.writer is not None:
            self.writer.add_scalar(key, preds, labels)
        self._data.total[key].extend(preds)
        self._data.counts[key].extend(labels)
        self._data.average[key] = average_precision_score(self._data.counts[key], self._data.total[key])

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    @property
    def data(self):
        return self._data
