import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(__file__)

def load_data(path):
    path = os.path.join(BASE_DIR, path)
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding = 'latin1')
    return data

def load_all_data():
    train = load_data('train')['data'].reshape(-1,3,32,32)
    test = load_data('test')['data'].reshape(-1,3,32,32)
    return train, test

class ImageDataset(Dataset):

    def __init__(self, data):
        self.data = torch.tensor(data)/255
        self.length = self.data.shape[0]

    def __getitem__(self, index):
        img = self.data[index]
        return img

    def __len__(self):
        return self.length

def make_dataloader(data):
    dataset = ImageDataset(data)
    return DataLoader(dataset, batch_size = 32, num_workers = 2)