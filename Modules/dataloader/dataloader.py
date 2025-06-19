import torch
from torch.utils.data import DataLoader

class ADNILoader(DataLoader):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.train_batch_size = 10
        self.eval_batch_size = 5
        self.train_shuffle = True
        self.eval_shuffle = False
        self.train_drop_last = False
        self.eval_drop_last = False
        self.num_workers = 20
        
    # def collate_fn(self, samples):
    #     pixel_values = torch.stack([sample[0] for sample in samples])
    #     labels = torch.tensor([sample[1] for sample in samples])
    #     return pixel_values, labels
        
    def train_dataloader(self):
        dataloader = DataLoader(self.kwargs['train_ds'],
                                batch_size=self.train_batch_size,
                                shuffle=self.train_shuffle,
                                num_workers=self.num_workers,
                                drop_last=self.train_drop_last,
                                # collate_fn = self.collate_fn
                               )
        return dataloader
    
    def validation_dataloader(self):
        dataloader = DataLoader(self.kwargs['valid_ds'],
                                batch_size=self.eval_batch_size,
                                shuffle=self.eval_shuffle,
                                num_workers=self.num_workers,
                                drop_last=self.eval_drop_last,
                                # collate_fn = self.collate_fn
                               )
        return dataloader
    
    def test_dataloader(self):
        dataloader = DataLoader(self.kwargs['test_ds'],
                                batch_size=self.eval_batch_size,
                                shuffle=self.eval_shuffle,
                                num_workers=self.num_workers,
                                drop_last=self.eval_drop_last,
                                # collate_fn = self.collate_fn
                               )
        return dataloader