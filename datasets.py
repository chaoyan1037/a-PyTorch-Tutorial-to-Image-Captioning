import json
import torch
from torch.utils.data import Dataset
import os


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_json):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        # Open data json file.
        with open(data_json, 'r') as f:
            data = json.load(f)
        
        self.caplens = []
        self.captions = []
        self.features = []
        for pid, record in data.items():
            self.caplens.append(record['cap_length'])
            self.captions.append(record['vector'])
            self.features.append(record['feature'])

        # Captions per image
        self.cpi = 1

        # Total number of datapoints
        self.dataset_size = len(self.caplens)

    def __getitem__(self, i):
        feat = torch.FloatTensor(self.features[i])
        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])
        return feat, caption, caplen
        
    def __len__(self):
        return self.dataset_size
