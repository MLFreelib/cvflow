from pathlib import Path
from random import random

import cv2
import math
import torchvision
from PIL import Image as Img
from torch.cuda import amp
from torchvision import transforms
from layers import *
from render import Detections, xywh2xyxy, xyxy2xywh, clip_coords
from torch.optim import SGD, Adam, AdamW, lr_scheduler
import numpy as np
from loss import ComputeLoss
import torch


def box_iou(box1, box2):
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]

    return output


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = 0
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except Exception:
        pass

    return s


def exif_transpose(image):
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {2: Img.FLIP_LEFT_RIGHT,
                  3: Img.ROTATE_180,
                  4: Img.FLIP_TOP_BOTTOM,
                  5: Img.TRANSPOSE,
                  6: Img.ROTATE_270,
                  7: Img.TRANSVERSE,
                  8: Img.ROTATE_90,
                  }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def smooth_BCE(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    #print('IM SHAPE BEF IN', im.shape)
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        im = torch.permute(im, (2, 0, 1))
        p = transforms.Resize((new_unpad[1], new_unpad[0]))
        im = p(im)
        im = torch.permute(im, (1, 2, 0))
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    pad = transforms.Compose([transforms.Pad([top, left, bottom, right], fill=0.44706)])
    im = torch.permute(im, (2, 1, 0))
    im = pad(im)
    im = torch.permute(im, (2, 1, 0))
    return im, ratio, (dw, dh)


class YOLO:
    def __init__(self, backbone, head):
        self.boxes = []
        self.pt = None
        self.backbone = backbone
        self.head = head
        self.stride = torch.tensor([8., 16., 32.])
        #self.model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
        self.outs = []
        self.head.bb_outs = self.backbone.outs

    def forward(self, x):
        init_x = np.copy(x)
        x = x.astype('float32')
        #print(x)
        #tx = transforms.ToTensor()(x).unsqueeze(0)
        #tx = letterbox(tx)[0]
        #tx = torch.nn.functional.pad(tx, init_p, value=0.44706)
        #print('IM INF', type(x), x.shape, x.dtype)
        tx = torch.from_numpy(x)
        #print(tx.shape)
        #tx = torch.permute(tx, (2, 0, 1))
        tx = tx.float()  # uint8 to fp16/32
        tx /= 255  # 0 - 255 to 0.0 - 1.0
        shape0 = tx.shape[:2]
        #print('TX SHAPE', tx.shape)
        tx = letterbox(tx)[0]
        #print('TTX SHAPE AFTER', tx.shape)
        shape1 = tx.shape[:2]
        #tx = tx.permute(2, 0, 1)
        #torch.set_printoptions(edgeitems=10)
        #tx = torch.nn.functional.pad(tx, init_p, value=114)
        #print(tx)
        torch.set_printoptions(profile="default")
        tx = tx[None]
        tx = torch.permute(tx, (0, 3, 1, 2))
        #print(tx)
        #tx, ratio, (dw, dh) = letterbox(x)
        #tx = transforms.ToTensor()(tx).unsqueeze(0)
        #with open('img.txt', 'w') as f:
        #    #print('WRITING')
        #    #f.write(str(tx))
        #print(tx.shape)
        with torch.no_grad():
            tx = self.backbone(tx)
        self.head.outs += self.backbone.outs
        self.backbone.outs.clear()
        with torch.no_grad():
            res = self.head(tx)
        self.head.outs.clear()
        #res_c = res[0].detach()
        res = non_max_suppression(res[0], 0.001, iou_thres=0.35, classes=None,
                                agnostic=False, multi_label=False, max_det=1000)  # NMS
        print(scale_coords(shape1, res[0][:, :4], shape0))
        #print('CV1 WEIGHT', self.backbone.conv1.layers[0].weight)
        #print('CV1 BIAS', self.backbone.conv1.layers[0].bias)
        #with amp.autocast(enabled=autocast):
        #    # Inference
        #    y = self.model(x, augment, profile)  # forward
        #    t.append(time_sync())
        #
        #            # Post-process
        #    y = non_max_suppression(y if self.dmb else y[0], self.conf, iou_thres=self.iou, classes=self.classes,
        #                            agnostic=self.agnostic, multi_label=self.multi_label, max_det=self.max_det)  # NMS
        #    for i in range(n):
        #        scale_coords(shape1, y[i][:, :4], shape0[i])
        #
        #    t.append(time_sync())
        #    return Detections(imgs, y, files, t, self.names, x.shape)

        #dets = Detections(init_x, res, None)
        #self.dets = dets
        self.res = res
        #self.boxes = res.xyxy[0]
        #self.boxes = res.xyxy
        self.boxes = res[0][..., :4]
        #self.boxes[:, [1, 3]] -= 3.48
        self.boxes /= 0.28
        self.scores = res[0][..., 4]
        self.labels = res[0][..., 5]
        out = [{'boxes': self.boxes,
                'labels': self.labels,
                'scores': self.scores},]
        return out

    def train(self, dataloader, save_dir=None, device='cpu', epochs=1):
        lr = 1e-4
        lrf = 1e-4
        momentum = 1e-2
        wu_bias_lr = 1e-4
        wu_momentum = 1e-4

        #print('VER', torch.__version__)
        scaler = amp.GradScaler(enabled=False)

        # Directories
        # save weights path
        # latest and best weights path
        if save_dir:
            w = save_dir / 'weights'  # weights dir
            last, best = w / 'last.pt', w / 'best.pt'

        # Loggers
        # TODO

        # optimizer
        g0, g1, g2 = [], [], []  # optimizer parameter groups
        '''for v in self.model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                g2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
                g0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g1.append(v.weight)
        #optimizer = Adam(g0, lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum'''
        del g0, g1, g2

        compute_loss = ComputeLoss(self.head)
        # Scheduler
        lf = lambda x: (1 - x / epochs) * (1.0 - lrf) + lrf
        #scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
        start_epoch = 0
        for epoch in range(start_epoch, epochs):
            mloss = torch.zeros(3, device=device)  # mean losses
            pbar = enumerate(dataloader)
            nw = 100
            #optimizer.zero_grad()
            nb = 0
            for i, (imgs, targets, paths, _) in pbar:
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(device, non_blocking=True).float() / 255
                # Warmup
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                    accumulate = max(1, np.interp(ni, xi, [1, 64 / 64]).round())
                    '''for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi,
                                            [wu_bias_lr if j == 2 else 0.0, x['initial_lr'] * 10])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [wu_momentum, momentum])'''

                # Forward
                with amp.autocast(enabled=False):
                    pred = self.forward(imgs)  # forward
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size

                scaler.scale(loss).backward()

                # Optimize
                '''if ni - last_opt_step >= accumulate:
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                    optimizer.zero_grad()

                    last_opt_step = ni'''

                mloss = (mloss * i + loss_items) / (i + 1)

            #scheduler.step()
