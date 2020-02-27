from torch.utils.data import Dataset
import torch
import json
from PIL import Image
import os


class trainloader(Dataset):
    def __init__(
        self,
        transform,
        base_path='../data/',
        meta_file='bbx_data/data_aug.json'
    ):

        self.base_path = base_path
        self.meta_path = os.path.join(self.base_path, meta_file)

        self.meta_data = json.load(open(self.meta_path))
        self.patient_ids = list(self.meta_data.keys())

        self.transform = transform


    def __getitem__(self, index):
        assert index <= len(self.patient_ids)
        patient_id = self.patient_ids[index]

        image_stack = []
        label_stack = []
        for frame in self.meta_data[patient_id]:
            frame_data = self.meta_data[patient_id][frame]
            image_path = os.path.join(self.base_path, frame_data['img_path'])
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
            frame_label = frame_data['label']

            image_stack.append(image)
            label_stack.append(frame_label)

        image_stack = torch.stack(image_stack, dim=0)

        return image_stack, label_stack

    def __len__(self):
        return len(self.patient_ids)
