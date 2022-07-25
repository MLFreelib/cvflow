import os
import shutil
import os.path as osp
import cv2
import copy
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import io, transform
from time import time
import math
import random
from IPython.display import clear_output
from tqdm import tqdm, tqdm_notebook
import seaborn as sns
import torch
from torchvision import models as mdl
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import threshold
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.nn import DataParallel
from torchvision.transforms import InterpolationMode
import torchvision.transforms as torT
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import mmcv
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
import torchvision
import mmdet
from mmcv import Config
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets import build_dataset, CocoDataset
from mmdet.apis import set_random_seed, train_detector
from mmdet.models.detectors import BaseDetector
from mmdet.apis import single_gpu_test, multi_gpu_test
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
import sys
from mmcls.apis import inference_model, init_model
import mmcls
from mmcls.models import build_classifier

root_path = 'Practice/'
classes = ('bus', 'car', 'cyclist', 'jeep',
           'misc', 'pedestrian', 'truck', 'van')
#device = 'cuda:0'
device = 'cpu'

class SuperModel():
    def __init__(self, classes, device):
        self.device = device
        self.classes = classes

    def _init_det_model(self, base_det_config):
        det_cfg, det_checkpoint, with_mask = base_det_config['cfg'], \
                                             base_det_config['checkpoint'], base_det_config['with_mask']

        self.with_mask = with_mask

        det_cfg = Config.fromfile(det_cfg)
        det_cfg.model.pretrained = None

        self.det_model = build_detector(det_cfg.model)

        checkpoint = load_checkpoint(self.det_model, det_checkpoint, map_location=self.device)

        self.det_model.CLASSES = checkpoint['meta']['CLASSES']
        self.det_model.cfg = det_cfg
        self.det_model.to(self.device)
        self.det_model.eval()

        self.det_model = init_detector(det_cfg, det_checkpoint, device=self.device)

        clear_output()

    def _init_cls_inference(self, base_cls_cfg, cls_checkpoint):
        self.cls_cfg = Config.fromfile(base_cls_cfg)
        self.cls_cfg.model.pretrained = None
        self.cls_model = build_classifier(self.cls_cfg.model)
        #checkpoint = load_checkpoint(self.cls_model, cls_checkpoint, map_location=device)
        self.cls_model.CLASSES = self.classes
        self.cls_model.cfg = self.cls_cfg
        self.cls_model.to(device)
        self.cls_model.eval()
        clear_output()

    def _init_cls_model(self, base_cls_cfg, pretrained=False):
        self.cls_cfg = Config.fromfile(base_cls_cfg)
        if pretrained: self.cls_cfg.resume_from = \
            root_path + 'logs/classifiers/' + \
            base_cls_cfg.split('/')[-2] + '/latest.pth'

        mmcls.apis.set_random_seed(0, deterministic=True)
        mmcv.mkdir_or_exist(osp.abspath(self.cls_cfg.work_dir))

        self.cls_model = build_classifier(self.cls_cfg.model)
        self.cls_model.cfg = self.cls_cfg
        self.cls_model.CLASSES = self.classes

        clear_output()

    def _train_classifier(self):
        self.cls_model.init_weights()
        datasets = [mmcls.datasets.build_dataset(self.cls_cfg.data.train),
                    mmcls.datasets.build_dataset(self.cls_cfg.data.test)]
        # Start fine-tuning
        mmcls.apis.train_model(
            self.cls_model,
            datasets[0],
            self.cls_cfg,
            distributed=False,
            validate=True)

    def create_bboxes(self, det_result, threshold):
        bboxes = np.array([])
        det_result = det_result[0] if self.with_mask else det_result
        for det_class in det_result:
            if det_class.any():
                for obj in det_class:
                    if obj[-1] > threshold:
                        bboxes = np.append(bboxes, obj[:-1])

        return bboxes

    def create_objects(self, img, bboxes):
        objects = []
        for box in bboxes:
            box = [min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])]
            img_res = img.crop(box)
            objects.append(mmcv.imread(np.array(img_res)))

        return objects

    def one_image_result(self, img, boxes, threshold=0.4):
        pil_img = Image.fromarray(img[0].astype(np.uint8).transpose(1, 2, 0))
        objects = self.create_objects(pil_img, boxes.cpu().detach().numpy().astype(np.uint8))
        labels = []
        for i in range(len(objects)):
            cls_result = inference_model(self.cls_model, objects[i])
            labels.append(cls_result['pred_label'])
        return labels
