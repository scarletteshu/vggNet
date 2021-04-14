from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import datasets, transforms
import numpy as np


class Dataloader:
    def __init__(self, cfgs, train_data_path:str, test_data_path:str):
        self.transform = transforms.Compose([
            #transforms.Resize(256),
            # #transforms.CenterCrop(224),
            transforms.Grayscale(1),
            transforms.ToTensor(),   #img to[0,1]
            transforms.Normalize(mean=[0.485, ], std=[0.229, ]),
        ])
        self.train_pth = train_data_path
        self.test_pth = test_data_path
        self.cfgs = cfgs

    def trDataLoader(self):
        train_dataset = datasets.ImageFolder(self.train_pth, transform=self.transform)
        # shuffle: to have the data reshuffled at every epoch. shuffle first then batch
        train_dataloader = DataLoader(train_dataset, batch_size= self.cfgs['batch_size'], shuffle=True)
        return train_dataset, train_dataloader

    def tstDataLoader(self):
        test_dataset = datasets.ImageFolder(self.test_pth, transform=self.transform)
        test_dataloader = DataLoader(test_dataset, batch_size= self.cfgs['batch_size'], shuffle=True)
        return test_dataset, test_dataloader