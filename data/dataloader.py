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
        self.mode = mode
        if mode == 'train':
            self.data_path = os.path.join(dataset_path, "Training")
            self.meta_path = os.path.join(dataset_path, 'train_meta.json')
        elif mode == 'test':
            self.data_path = os.path.join(dataset_path, "Testing")
            self.meta_path = os.path.join(dataset_path, 'test_meta.json')
        else:
            self.data_path = os.path.join(dataset_path, "Training")
            self.meta_path = os.path.join(dataset_path, 'val_meta.json')

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

        # print("get item")
        # print(id_data)

        if label == 1 and self.mode == "train":
            bbox = id_data["bbox"]
            x1 = round(float(bbox["left"]))
            x2 = round(float(bbox["right"]))
            y1 = round(float(bbox["upper"]))
            y2 = round(float(bbox["lower"]))

        for frame in id_data["sequence"]:
            image_path = self.data_path + id_data["sequence"][frame]
            image = Image.open(image_path).convert("RGB")
            if label == 1 and self.mode == "train":
                image.crop((x1, y1, x2, y2))
            image = self.transform(image)
            image_stack.append(image)

        if self.pad_sequence:
            image_stack.extend(
                [zero_img] * (self.seq_length - len(image_stack)))

        image_stack = torch.stack(image_stack, dim=0)
        label = torch.Tensor([label])
        # label = label.squeeze(-1)


        return image_stack, label

    def __len__(self):
        return len(self.indexes)
