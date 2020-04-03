from torch.utils.data import Dataset
import torch
import json
from PIL import Image
import os
import numpy as np


class DataLoader(Dataset):
    def __init__(
        self,
        transform,
        mode="train",
        dataset_path='/srv/home/deepandas11/bleeds/data/Data/DataSet13_20200221/raw_patient_based',
    ):
        if mode == 'train':
            self.data_path = os.path.join(dataset_path, "Training")
            self.meta_path = os.path.join(dataset_path, 'train_meta.json')
        else:
            self.data_path = os.path.join(dataset_path, "Testing")
            self.meta_path = os.path.join(dataset_path, 'test_meta.json')
        self.meta_data = json.load(open(self.meta_path, 'r'))
        self.indexes = list(self.meta_data.keys())
        self.transform = transform

    def __getitem__(self, index):
        assert index <= len(self.indexes)
        pid = self.indexes[index]
        id_data = self.meta_data[pid]

        image_stack = []
        label = id_data["seq_label"]

        for frame in id_data["sequence"]:
            # rel_path = os.path.join(self.data_path, id_data["sequence"][frame])
            image_path = self.data_path + id_data["sequence"][frame]
            # print("hey ho", image_path)
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
            image_stack.append(image)

        image_stack = torch.stack(image_stack, dim=0)
        label = torch.Tensor([label]).unsqueeze(0)

        return image_stack, label

    def __len__(self):
        return len(self.indexes)

    def get_indices(self):
        return np.random.permutation(np.arange(len(self.indexes)))
