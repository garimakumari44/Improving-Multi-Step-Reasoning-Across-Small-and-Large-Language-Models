import torch
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    def __init__(self, size=1000, input_dim=128, num_classes=10):
        self.x = torch.randn(size, input_dim)
        self.y = torch.randint(0, num_classes, (size,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]