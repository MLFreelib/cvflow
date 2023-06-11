import os
import re
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision.transforms.functional as F
from Meta import MetaName, MetaDepth


class StereoDataset(Dataset):

    def __init__(self, data_folder, split, is_left = True ):
        super().__init__()
        self.is_left = is_left

        self.data_folder = data_folder
        if self.is_left:
            self.filenames = os.listdir(os.path.join(data_folder, 'frames_cleanpass', 'left'))
        else:
            self.filenames = os.listdir(os.path.join(data_folder, 'frames_cleanpass', 'right'))
        self.disp_filenames = os.listdir(os.path.join(data_folder, 'disparity'))

        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST', 'VALID', 'TEMPLATES'}

        assert len(self.filenames)  ==len(self.disp_filenames)

    def __len__(self):
        return len(self.filenames)
    def load_image(self, filename):
        return Image.open(filename).convert('RGB')
    def load_disp(self, filename):
        data, scale = self.pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def pfm_imread(self, filename):
        file = open(filename, 'rb')
        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip())
        if scale < 0:
            endian = '<'
            scale = -scale
        else:
            endian = '>'

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale
    #
    def __getitem__(self, index):
        if self.is_left:
            img = self.load_image(os.path.join(self.data_folder, 'frames_cleanpass', 'left', self.filenames[index]))
        else:
            img = self.load_image(os.path.join(self.data_folder, 'frames_cleanpass', 'right', self.filenames[index]))

        disparity = self.load_disp(os.path.join(self.data_folder,'disparity', self.disp_filenames[index]))
        w, h = img.size
        crop_w, crop_h = 512, 256

        x1 = random.randint(0, w - crop_w)
        y1 = random.randint(0, h - crop_h)

        # random crop
        img = F.to_tensor(img.resize((crop_w, crop_h)))
        disparity = torch.from_numpy(cv2.resize(disparity, (crop_w, crop_h)))


        disparity = MetaDepth(depth=disparity)
        return {"image": img,
                MetaName.META_DEPTH.value: disparity}