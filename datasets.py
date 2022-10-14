import matplotlib.pyplot as plt
import random
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

DEFAULT_LIVESTOCK_DIR = "./data/livestock/part_III_cropped"

class LivestockTrainDataset(Dataset):
    def __init__(self, img_size, fake_dataset_size):
        if os.path.isdir(DEFAULT_LIVESTOCK_DIR):
            self.img_dir = os.path.join(DEFAULT_LIVESTOCK_DIR, "Train")
        else:
            self.img_dir = UNDEFINE
        self.img_files = list(
                            np.random.choice(
                                [os.path.join(self.img_dir, img)
                            for img in os.listdir(self.img_dir)
                            if (os.path.isfile(os.path.join(self.img_dir,
                            img)) and img.endswith('jpg'))],
                            size=fake_dataset_size)
                            )
        self.fake_dataset_size = fake_dataset_size # needed otherwise there are
        # 125000 images, and this is too much
        self.transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.PILToTensor(),
            transforms.Lambda(lambda img: img.float()),
            transforms.Lambda(lambda img: img / 255.)
        ]) 
        self.nb_img = len(self.img_files)
        self.nb_channels = 3

    def __len__(self):
        return max(self.nb_img, self.fake_dataset_size)

    def __getitem__(self, index):
        index = index % self.nb_img
        img = Image.open(self.img_files[index])
        
        return self.transform(img), 1 # one if the ground truth if there is one

class LivestockTestDataset(Dataset):
    def __init__(self, img_size, fake_dataset_size):
        if os.path.isdir(DEFAULT_LIVESTOCK_DIR):
            self.img_dir = os.path.join(DEFAULT_LIVESTOCK_DIR, "Test")
        else:
            self.img_dir = UNDEFINE
        self.img_files = list(
                            np.random.choice(
                                [os.path.join(self.img_dir, img)
                            for img in os.listdir(self.img_dir)
                            if (os.path.isfile(os.path.join(self.img_dir, img))
                            and img.endswith('.jpg'))],
                            size=fake_dataset_size)
                            )
        self.fake_dataset_size = fake_dataset_size # needed otherwise there are
        self.gt_files = [s.replace(".jpg", "_gt.png") for s in self.img_files]
        self.transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.PILToTensor(),
            transforms.Lambda(lambda img: img.float()),
            transforms.Lambda(lambda img: img / 255.)
        ]) 
        self.nb_img = len(self.img_files) # recompute the size,
        # fake_dataset_size may have changed it
        self.nb_channels = 3

    def __len__(self):
        return self.fake_dataset_size

    def __getitem__(self, index):
        img = Image.open(self.img_files[index])
        gt = Image.open(self.gt_files[index])

        return self.transform(img), self.transform(gt)

