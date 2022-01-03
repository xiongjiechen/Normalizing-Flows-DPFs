import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

class ToyDiskDataset(Dataset):
    def __init__(self, data_path, filename, datatype="train_data"):
        # datatype: train_data, val_data, test_data
        self.data_path=data_path
        self.filename=filename

        files = os.listdir(self.data_path)

        self.train_files = \
            [os.path.join(self.data_path, f) for f in files
             if f.startswith(self.filename) and
             'train' in f]
        self.val_files = \
            [os.path.join(self.data_path, f) for f in files
             if f.startswith(self.filename) and
             'val' in f]
        self.test_files = \
            [os.path.join(self.data_path, f) for f in files
             if f.startswith(self.filename) and
             'test' in f]

        self.train_data=sorted(self.train_files)
        self.val_data=sorted(self.val_files)
        self.test_data=sorted(self.test_files)

        if datatype=="train_data":
            loadData=self.train_data
        elif datatype=="val_data":
            loadData = self.val_data
        else:
            loadData = self.test_data

        for index in range(1): # previous: 3 for index in range(3):; for semi: change it to 1 to obtain 1/3
            data = dict(np.load(loadData[index], allow_pickle=True))[datatype].item()
            if index == 0:
                self.start_image = data['start_image']
                self.start_state = data['start_state']
                self.image = data['image']
                self.state = data['state']
                self.q = data['q']
                self.visible = data['visible']
            else:
                self.start_image = np.concatenate((self.start_image, data['start_image']), axis=0)
                self.start_state = np.concatenate((self.start_state, data['start_state']), axis=0)
                self.image = np.concatenate((self.image, data['image']), axis=0)
                self.state = np.concatenate((self.state, data['state']), axis=0)
                self.q = np.concatenate((self.q, data['q']), axis=0)
                self.visible = np.concatenate((self.visible, data['visible']), axis=0)

        self.data_size = len(self.start_image)

        # debug
        print(self.data_size)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self.start_image[idx], self.start_state[idx], self.image[idx], self.state[idx], self.q[idx], self.visible[idx])
