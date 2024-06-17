import matplotlib.pyplot as plt
import random
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

DEFAULT_MVTEC_DIR = "../data/mvtec_anomaly_detection/"
DEFAULT_LIVESTOCK_DIR = "../data/livestock/part_III_cropped"

class MVTEC_train_Dataset(Dataset):
    def __init__(self, img_size, category, fake_dataset_size, transform=None,
        target_transform=None):
        self.img_dir = os.path.join(DEFAULT_MVTEC_DIR, category, "train/good/")
        self.img_files = [os.path.join(self.img_dir, img)
                          for img in os.listdir(self.img_dir)
                          if os.path.isfile(os.path.join(self.img_dir, img))]
        self.img_size = img_size
        if category in ['hazelnut', 'bottle', 'metal_nut', 'screw']:
            self.transform = transforms.Compose([
                transforms.RandomRotation(degrees=45),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(size=(img_size, img_size)),
            ]) 
        elif category in ['toothbrush', 'transistor']:
            self.transform = transforms.Compose([
                transforms.RandomRotation(degrees=5),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(size=(img_size, img_size)),
            ]) 
        elif category in ['capsule', 'zipper']:
            self.transform = transforms.Compose([
                transforms.RandomRotation(degrees=5),
                transforms.RandomVerticalFlip(),
                transforms.Resize(size=(img_size, img_size)),
            ]) 
        elif category in ['cable', 'pill']:
            self.transform = transforms.Compose([
                transforms.RandomRotation(degrees=5),
                transforms.Resize(size=(img_size, img_size)),
            ]) 
        elif category in ['wood', 'leather', 'grid', 'carpet', 'tile']: # textures
            self.transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=(img_size, img_size),
                    pad_if_needed=True, padding_mode="symmetric"),
            ]) 
        else:
            raise RuntimeError("Bad category")
        
        self.img_size = img_size
        self.target_transform = target_transform
        self.fake_dataset_size = fake_dataset_size
        self.nb_img = len(self.img_files)
        self.nb_channels = 3

    def __len__(self):
        return max(self.nb_img, self.fake_dataset_size)

    def __getitem__(self, index):
        index = index % self.nb_img
        img = Image.open(self.img_files[index])
        np_img = np.asarray(img)
        if np_img.ndim == 2:
            np_img = np.stack([np_img for i in range(3)], axis=2)
        img = Image.fromarray(np_img)
        out = self.transform(img)
        out = transforms.ToTensor()(out)
        return out, 1 # one if the ground truth if there is one

class MVTEC_test_Dataset(Dataset):
    def __init__(self, img_size, category, defect,
        transform=None, target_transform=None):
        if defect is not None:
            self.img_dir = os.path.join(DEFAULT_MVTEC_DIR, category, "test", defect)
        else:
            defects = os.listdir(os.path.join(DEFAULT_MVTEC_DIR, category, "test"))
            defect = defects[0]
            self.img_dir = os.path.join(DEFAULT_MVTEC_DIR, category, "test", defect)

        self.img_files = [os.path.join(self.img_dir, img)
                          for img in os.listdir(self.img_dir)
                          if os.path.isfile(os.path.join(self.img_dir, img))]
        self.gt_dir = os.path.join(DEFAULT_MVTEC_DIR, category, "ground_truth", defect)
        if defect != "good":
            self.gt_files = [os.path.join(self.gt_dir, n[-7:-4] + "_mask.png")
                         for n in self.img_files]
        else:
            self.gt_files = []
        self.category = category
        self.ori_size = 1024
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize(size=(self.img_size, self.img_size)),
            transforms.ToTensor(),
        ]) 
        self.target_transform = transforms.Compose([
            transforms.Resize(size=(self.img_size, self.img_size)),
            transforms.ToTensor(),
        ]) 
        self.img_size = img_size
        self.nb_channels = 3

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):

        img = Image.open(self.img_files[index])
        np_img = np.asarray(img)
        if np_img.ndim == 2:
            np_img = np.stack([np_img for i in range(3)], axis=2)
        img = Image.fromarray(np_img)
        transformed_img = self.transform(img)

        if self.gt_files:
            gt = Image.open(self.gt_files[index])
            transformed_gt = self.target_transform(gt)
        else:
            transformed_gt = torch.zeros(transformed_img.shape)

        return transformed_img, transformed_gt

class LivestockTrainDataset(Dataset):
    def __init__(self, img_size, fake_dataset_size):
        if os.path.isdir(DEFAULT_LIVESTOCK_DIR):
            self.img_dir = os.path.join(DEFAULT_LIVESTOCK_DIR, "Train")
        else:
            self.img_dir = UNDEFINE
        self.img_files = list(
                            np.random.choice([os.path.join(self.img_dir, img)
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
        return self.nb_img

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

