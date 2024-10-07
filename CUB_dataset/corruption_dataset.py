from torch.utils.data import Dataset
from .corruption import corruptions
import random
from .utils import restore_image_from_numpy, open_image
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import os
import csv


class Corruption_dataset(Dataset):

    def __init__(self, dataset, transform, prob=15/16):
        self.dataset = dataset
        self.transform = transform
        self.prob = prob

        self.corruptions = [
            "gaussian_noise", "shot_noise", "impulse_noise",
            "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
            "snow", "frost", "fog", "brightness", 
            "contrast", "elastic_transform", "pixelate", "jpeg_compression", 
        ]
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        if random.random() < self.prob:
            crp_func = random.choice(self.corruptions)
            sev = random.choice([1, 2, 3, 4, 5])
            crp_func = corruptions[crp_func]
            distorted_image = crp_func(image, sev)
            distorted_image = restore_image_from_numpy(distorted_image)
        else:
            distorted_image = image

        image = self.transform(image)
        distorted_image = self.transform(distorted_image)
        
        return distorted_image, label


class Val_Corruption_cub(Dataset):

    def __init__(self, corruption_root, csv_path, transform):
        self.corruption_root = corruption_root
        self.csv_path = csv_path
        self.transform = transform

        self.images = []
        self._load_images()
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        imagepath, label = self.images[idx]
        image = open_image(imagepath)
        image = self.transform(image)

        return image, label

    def _load_images(self):
        for row in self._read_csv():
            imagepath = os.path.join(self.corruption_root, row[0])
            label = int(row[2]) - 1
            self.images.append((imagepath, label))

    def _read_csv(self):
        with open(self.csv_path) as csv_file:
            csv_file_rows = csv.reader(csv_file, delimiter=",")
            for row in csv_file_rows:
                yield row