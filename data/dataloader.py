from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import json
from PIL import Image
import os


def get_loader(dataset, batch_size=8, shuffle=True):
    loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


class BleedsDataset(Dataset):
    def __init__(
        self,
        transform,
        dataset_path,
        batch_size=8,
        upsample=True,
        pad_sequence=True,
        mode="train",
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
        self.upsample = upsample
        # Hardcoded max-length value
        self.seq_length = 38
        self.pad_sequence = pad_sequence

        if self.upsample and mode == 'train':
            # Upsampling ratio of 5 for this dataset
            self.indexes = self.indexes[:70] + self.indexes[70:]*5

    def __getitem__(self, index):
        assert index <= len(self.indexes)
        pid = self.indexes[index]
        id_data = self.meta_data[pid]
        zero_img = torch.zeros(3, 224, 224)
        image_stack = []
        label = id_data["seq_label"]

        for frame in id_data["sequence"]:
            image_path = self.data_path + id_data["sequence"][frame]
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
            image_stack.append(image)

        if self.pad_sequence:
            image_stack.extend(
                [zero_img] * (self.seq_length - len(image_stack)))

        image_stack = torch.stack(image_stack, dim=0)
        label = torch.Tensor([label]).long()
        label = label.squeeze(-1)


        return image_stack, label

    def __len__(self):
        return len(self.indexes)
