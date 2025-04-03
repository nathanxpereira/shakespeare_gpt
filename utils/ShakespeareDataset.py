import torch
from torch.utils.data import IterableDataset

class ShakespeareDataset(IterableDataset):
    def __init__(self, data: torch.Tensor, batch_size, block_size):
        self.data = data
        self.batch_size = batch_size
        self.block_size = block_size
        self.num_examples = len(data)-block_size-1

    def __len__(self):
        return self.num_examples//self.batch_size
    
    def __iter__(self):
        for _ in range(self.__len__()):
            indices = torch.randint(0, self.num_examples, (self.batch_size,))
            x = torch.stack([self.data[idx:idx+self.block_size] for idx in indices]).to(torch.int64)
            y = torch.stack([self.data[idx+1:idx+self.block_size+1] for idx in indices]).to(torch.int64)
            yield x,y
        