from torch.utils.data import DataLoader
from data.dataset import DummyDataset

def get_loader(batch_size=32):
    dataset = DummyDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)