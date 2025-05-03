#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: test_dataloader_rcan.py
# Created Date: Tuesday January 12th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 20th April 2023 9:31:19 am
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################

import os
import glob
from tqdm import tqdm
import cv2
import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms as T


class TestDataset:
    def __init__(self,
                 dataset_name,
                 data_root,
                 batch_size=16,
                 degradation="bicubic",
                 image_scale=4,
                 subffix='png'):
        """Initialize and preprocess the B100 dataset."""
        self.data_root = data_root
        self.image_scale = image_scale
        self.dataset_name = dataset_name
        self.subffix = subffix
        self.dataset = []
        self.pointer = 0
        self.batch_size = batch_size
        self.__preprocess__()
        self.num_images = len(self.dataset)

        if self.dataset_name.lower() == "set5":
            self.dataset_name = "Set5"
        elif self.dataset_name.lower() == "ixi":
            self.dataset_name = "ixi"
        elif self.dataset_name.lower() == "set14":
            self.dataset_name = "Set14"
        elif self.dataset_name.lower() == "b100":
            self.dataset_name = "B100"
        elif self.dataset_name.lower() == "urban100":
            self.dataset_name = "Urban100"

        # c_transforms = []
        # c_transforms.append(T.ToTensor())
        # c_transforms.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        # self.img_transform = T.Compose(c_transforms)

    def __preprocess__(self):
        """Preprocess the Urban100 dataset."""
        hr_path = os.path.join(self.data_root, "test_HR_T2")
        lr_path = os.path.join(self.data_root, "test_LR_T2", "X%d" % self.image_scale)
        ref_path = os.path.join(self.data_root, "test_HR_PD")

        print("Evaluation dataset HR path: %s" % hr_path)
        print("Evaluation dataset LR path: %s" % lr_path)
        assert os.path.exists(hr_path)
        assert os.path.exists(lr_path)
        assert os.path.exists(ref_path)
        hr_files = sorted(os.listdir(hr_path))
        lr_files = sorted(os.listdir(lr_path))
        ref_files = sorted(os.listdir(ref_path))
        hr_file_paths = [os.path.join(hr_path, i) for i in hr_files]
        lr_file_paths = [os.path.join(lr_path, i) for i in lr_files]
        ref_file_paths = [os.path.join(ref_path, i) for i in ref_files]
        self.filenames = lr_file_paths

        print("processing %s images..." % self.dataset_name)

        for idx in tqdm(range(len(lr_file_paths[:50]))):
            ref = cv2.imread(ref_file_paths[idx], cv2.IMREAD_UNCHANGED)
            hr_t2 = cv2.imread(hr_file_paths[idx], cv2.IMREAD_UNCHANGED)
            lr_t2 = cv2.imread(lr_file_paths[idx], cv2.IMREAD_UNCHANGED)
            # mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            ref = torch.tensor(ref).unsqueeze(0).float() / 255.
            hr_t2 = torch.tensor(hr_t2).unsqueeze(0).float() / 255.
            lr_t2 = torch.tensor(lr_t2).unsqueeze(0).float() / 255.
            self.dataset.append((hr_t2, lr_t2, ref))

        print('Finished preprocessing the ixi Validation dataset, total image number: %d...' % len(self.dataset))

    def __call__(self):
        """Return one batch images."""
        if self.pointer >= self.num_images:
            self.pointer = 0
            a = "The end of the story!"
            raise StopIteration(print(a))

        hr = self.dataset[self.pointer][0]
        # image = Image.open(filename)
        # hr = self.img_transform(image)
        lr = self.dataset[self.pointer][1]
        ref = self.dataset[self.pointer][2]
        # image = Image.open(filename)
        # lr = self.img_transform(image)
        file_name = os.path.basename(self.filenames[self.pointer])
        file_name = os.path.splitext(file_name)[0]
        hr_ls = hr.unsqueeze(0)
        lr_ls = lr.unsqueeze(0)
        ref = ref.unsqueeze(0)
        nm_ls = [file_name, ]

        self.pointer += 1
        return hr_ls, lr_ls,ref, nm_ls

    def __len__(self):
        return self.num_images

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.data_root + ')'
