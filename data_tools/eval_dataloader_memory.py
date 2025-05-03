#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: eval_dataloader_memory.py
# Created Date: Thursday March 11th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 20th April 2023 9:27:26 am
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################


import os
import cv2
import glob
import torch
from tqdm import tqdm


class EvalDataset:
    def __init__(self,
                 dataset_name,
                 data_root,
                 batch_size=1,
                 degradation="bicubic",
                 image_scale=4,
                 subffix='png',
                 dataloader_num='None'):
        """Initialize and preprocess the urban100 dataset."""
        self.data_root = data_root
        self.degradation = degradation
        self.image_scale = image_scale
        self.dataset_name = dataset_name
        self.subffix = subffix
        self.dataset = []
        self.pointer = 0
        self.batch_size = 1

        if self.dataset_name.lower() == "ixi":
            self.dataset_name = "ixi"
        elif self.dataset_name.lower() == "set14":
            self.dataset_name = "Set14"
        elif self.dataset_name.lower() == "b100":
            self.dataset_name = "B100"
        elif self.dataset_name.lower() == "urban100":
            self.dataset_name = "Urban100"
        else:
            raise FileNotFoundError
        print("%s dataset is used!" % self.dataset_name)
        self.dataloader_num = dataloader_num
        self.__preprocess__()
        self.num_images = len(self.dataset)

        # c_transforms  = []
        # c_transforms.append(T.ToTensor())
        # c_transforms.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        # self.img_transform = T.Compose(c_transforms)


    def __preprocess__(self):
        """Preprocess the Urban100 dataset."""
        hr_path = os.path.join(self.data_root, "val_HR_T2")
        lr_path = os.path.join(self.data_root, "val_LR_T2", "X%d"% self.image_scale)
        ref_path = os.path.join(self.data_root, "val_HR_PD")

        print("Evaluation dataset HR path: %s" % hr_path)
        print("Evaluation dataset LR path: %s" % lr_path)
        assert os.path.exists(hr_path)
        assert os.path.exists(lr_path)
        assert os.path.exists(ref_path)
        if self.dataloader_num == 'None':
            hr_files = sorted(os.listdir(hr_path))
            lr_files = sorted(os.listdir(lr_path))
            ref_files = sorted(os.listdir(ref_path))
        else:
            hr_files = sorted(os.listdir(hr_path))[:self.dataloader_num]
            lr_files = sorted(os.listdir(lr_path))[:self.dataloader_num]
            ref_files = sorted(os.listdir(ref_path))[:self.dataloader_num]

        hr_file_paths = [os.path.join(hr_path, i) for i in hr_files]
        lr_file_paths = [os.path.join(lr_path, i) for i in lr_files]
        ref_file_paths = [os.path.join(ref_path, i) for i in ref_files]

        print("processing %s images..." % self.dataset_name)


        for idx in tqdm(range(len(lr_file_paths))):
            ref = cv2.imread(ref_file_paths[idx], cv2.IMREAD_UNCHANGED)
            hr_t2 = cv2.imread(hr_file_paths[idx], cv2.IMREAD_UNCHANGED)
            lr_t2 = cv2.imread(lr_file_paths[idx], cv2.IMREAD_UNCHANGED)
            # mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            ref = torch.tensor(ref).unsqueeze(0).float() / 255.
            hr_t2 = torch.tensor(hr_t2).unsqueeze(0).float() / 255.
            lr_t2 = torch.tensor(lr_t2).unsqueeze(0).float() / 255.
            self.dataset.append((hr_t2,lr_t2,ref))

        print('Finished preprocessing the ixi Validation dataset, total image number: %d...' % len(self.dataset))

    def __call__(self):
        """Return one batch images."""
        if self.pointer >= self.num_images:
            self.pointer = 0
        hr = self.dataset[self.pointer][0]
        lr = self.dataset[self.pointer][1]
        ref = self.dataset[self.pointer][2]
        # hr = (hr / 255.0 - 0.5) * 2.0
        # lr = (lr / 255.0 - 0.5) * 2.0
        hr = hr.unsqueeze(0)
        lr = lr.unsqueeze(0)
        ref = ref.unsqueeze(0)
        self.pointer += 1
        return hr, lr,ref

    def __len__(self):
        return self.num_images

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.data_root + ')'
