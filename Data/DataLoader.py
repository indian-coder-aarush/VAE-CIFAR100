import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np

def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

train = load_data('train')['data']
test = load_data('test')['data']

train.reshape(-1,3,32,32)
test.reshape(-1.3,32,32)

class ImageDataset(Dataset):

    def __init__(self, data):
        self.data = torch.tensor(data)/255
        self.length = self.data.shape[0]

    def __getitem__(self, index):
        img = self.data[index]

    def __len__(self):
        return self.length

train_dataset = ImageDataset(train)
test_dataset = ImageDataset(test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers = 2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers = 2)