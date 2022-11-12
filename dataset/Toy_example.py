import torch
import numpy as np


class Toy_example(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        # array = (self.X[index])
        return self.X[index].unsqueeze(0).unsqueeze(0).unsqueeze(0), self.y[index]

    def __len__(self):
        return self.len
