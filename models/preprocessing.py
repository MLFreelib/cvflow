import numpy as np
import torch
import math

from torchvision import transforms

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
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

def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

def preprocess_for_YOLO(im, stride, size=640):
    shape0, shape1 = [], []
    x = torch.tensor([])
    if im.shape[0] < 5:  # image in CHW
        im = torch.permute(im[0], (1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
    s = im.shape[:2]  # HWC
    shape0.append(s)  # image shape
    g = (size / max(s))  # gain
    shape1.append([y * g for y in s])
    im = im  # if im.data.contiguous else np.ascontiguousarray(im)  # update
    x = torch.cat((x, letterbox(im)[0].unsqueeze(0)))
    #shape1 = [make_divisible(x, stride) for x in np.stack(shape1, 0).max(0)]  # inference shape
    x = torch.permute(x, (0, 3, 1, 2))
    print('XOUT', x.shape)
    return x