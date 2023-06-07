from torch.utils.data import Dataset, DataLoader

class FireDataloader():
    def __init__(self, dataset: Dataset = None, batch_size: int = 128, shuffle: bool = True, num_workers: int = 16,
                 pin_memory: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def dataloader(self):
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)
