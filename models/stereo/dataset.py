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

    def __init__(self, data_folder, split):
        super().__init__()

        self.data_folder = data_folder

        self.left_filenames = os.listdir(os.path.join(data_folder, 'frames_cleanpass', 'left'))
        self.right_filenames = os.listdir(os.path.join(data_folder, 'frames_cleanpass', 'right'))
        self.disp_filenames = os.listdir(os.path.join(data_folder, 'disparity'))

        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST', 'VALID', 'TEMPLATES'}

        assert len(self.left_filenames) == len(self.right_filenames) ==len(self.disp_filenames)

    def __len__(self):
        return len(self.left_filenames)
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
        left_img = self.load_image(os.path.join(self.data_folder, 'frames_cleanpass', 'left', self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.data_folder, 'frames_cleanpass', 'right', self.right_filenames[index]))
        disparity = self.load_disp(os.path.join(self.data_folder,'disparity', self.disp_filenames[index]))

        w, h = left_img.size
        crop_w, crop_h = 512, 256

        x1 = random.randint(0, w - crop_w)
        y1 = random.randint(0, h - crop_h)

        # random crop
        left_img = F.to_tensor(left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h)))
        right_img = F.to_tensor(right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h)))
        disparity = F.to_tensor(disparity[y1:y1 + crop_h, x1:x1 + crop_w])

        # to tensor, normalize
        # processed = self.get_transform()
        # left_img = processed(left_img)
        # right_img = processed(right_img)

        disparity = MetaDepth(depth=disparity)

        return {"image": left_img,
                MetaName.META_DEPTH.value: disparity}