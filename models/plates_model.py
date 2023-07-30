import sys

sys.path.append('../')
import torch
import torch.nn as nn
from PIL import Image
from models.blocks import CRNN
from common.ctc_decoder import ctc_decode
import numpy as np
from torchvision.transforms.functional import crop, resize, rgb_to_grayscale
from models.models import yolo_small
import cv2
import matplotlib.pyplot as plt

CHARS = '0123456789АВЕКМНОРСТУХ'
CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}


class PlatesModel(nn.Module):
    def __init__(self, yolo_checkpoint=None, crnn_checkpoint=None):
        super(PlatesModel, self).__init__()
        self.num_class = len(LABEL2CHAR) + 1
        self.img_width = 256
        self.img_height = 64

        # self.yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_checkpoint)  # local model
        self.yolo_model = yolo_small(weights_path=yolo_checkpoint)

        self.crnn = CRNN(in_channels=1, out_channels=None, img_height=self.img_height, img_width=self.img_width,
                         num_class=self.num_class)

        self.device = 'cuda' if next(self.crnn.parameters()).is_cuda else 'cpu'
        if crnn_checkpoint:
            self.crnn.load_state_dict(torch.load(crnn_checkpoint, map_location=self.device))
        self.crnn.to(self.device)

    def forward(self, imgs):
        """
        :param imgs: tensor [N, 3, W, H]
        """
        det = self.yolo_model(imgs)
        imgs = rgb_to_grayscale(imgs)
        for i in range(len(imgs)):
            img = imgs[i]  # torch tensor for i-th image
            bbset = det[i]['boxes']
            zero_mask = (bbset[:, [0, 2]] == .0).all(dim=1) | (bbset[:, [1, 3]] == .0).all(dim=1) | \
                        (bbset[:, 1] > bbset[:, 3]) | (bbset[:, 0] > bbset[:, 2])
            conf = det[i]['scores'][~zero_mask]
            bbset = bbset[~zero_mask]

            true_conf = conf > 0.25

            conf = conf[true_conf]
            bbset = bbset[true_conf]

            img_size = img.size()[1:]
            platesset = []  # set of plates for a single frame
            correct_indexes = []
            for j, bbox in enumerate(bbset):
                x, y, x2, y2 = bbox.tolist()
                h = int((y2 - y) * img_size[0])
                w = int((x2 - x) * img_size[1])
                x = int(x * img_size[1])
                y = int(y * img_size[0])
                if w == 0 or h == 0:
                    continue
                correct_indexes.append(j)
                cr = crop(img, y, x, h, w)
                cr = resize(cr, [self.img_height, self.img_width]).unsqueeze(0)
                # cr = (cr / 127.5) - 1.0
                # cr_save = cr.permute(1, 2, 0).cpu().detach().numpy()
                # plt.imsave('temp.png', cr_save)
                with torch.no_grad():
                    cr = cr.to(self.device)
                    logits = self.crnn(cr)
                    log_probs = torch.nn.functional.log_softmax(logits, dim=2)
                    preds = ctc_decode(log_probs, method='beam_search', beam_size=10,
                                       label2char=LABEL2CHAR)
                    platesset.append(''.join(preds[0]))

            det[i]['labels'] = platesset
            det[i]['scores'] = conf[correct_indexes]
            det[i]['boxes'] = bbset[correct_indexes]

        return det
