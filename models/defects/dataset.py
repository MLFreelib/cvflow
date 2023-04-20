import os

import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from models.defects.utils import box2imgsize_box, transform


class GetterBase(Dataset):
    def __init__(self, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """

        self.keep_difficult = keep_difficult

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties


class CustomDataset(GetterBase):

    def __init__(self, data_folder, split, anno_postfix: str, img_postfix: str, keep_difficult=False,
                 label_aux: int = 0):
        super().__init__(data_folder)

        self.split = split.upper()
        self.anno_postfix = anno_postfix
        self.img_postfix = img_postfix
        assert self.split in {'TRAIN', 'TEST', 'VALID', 'TEMPLATES'}
        self.label_aux = label_aux

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        self.le = LabelEncoder()
        self.get_classes(data_folder)
        self.objects, self.anno_list = self.get_objects(data_folder)
        self.images = self.get_images()

        assert len(self.images) == len(self.objects)

    def get_images(self):
        images = list()
        for anno in self.anno_list:
            img_path = self.__anno2img_path(anno)
            if os.path.exists(img_path):
                images.append(img_path)
        return images

    def get_classes(self, path: str):
        anno_path = os.path.join(path, 'bboxes')
        annos = os.listdir(anno_path)
        labels = list()
        for anno in annos:
            full_anno_path = os.path.join(anno_path, anno)
            img_path = self.__anno2img_path(full_anno_path)
            if not os.path.exists(img_path):
                continue
            with open(full_anno_path, 'r') as f:

                for line in f.readlines():
                    info = line.strip().split()
                    labels.append(info[0])
        self.le.fit(labels)

    def get_objects(self, path: str):
        anno_path = os.path.join(path, 'bboxes')
        annos = os.listdir(anno_path)
        objects = list()
        annos_list = list()
        for anno in annos:
            full_anno_path = os.path.join(anno_path, anno)
            img_path = self.__anno2img_path(full_anno_path)
            if not os.path.exists(img_path):
                continue
            with open(full_anno_path, 'r') as f:
                boxes = list()
                labels = list()
                difficulties = list()

                for line in f.readlines():
                    info = line.strip().split()
                    box = [float(box.replace(',', '.')) for box in info[1:]]
                    box = box2imgsize_box(img_path, box)
                    boxes.append(box)
                    labels.append(info[0])
                    difficulties.append(0)
                if len(boxes) != 0:
                    labels = np.array(self.le.transform(labels))
                    labels = (labels + self.label_aux).tolist()
                    objects.append({'boxes': boxes, 'labels': labels, 'difficulties': difficulties})
                    annos_list.append(full_anno_path)

        return objects, annos_list

    def __anno2img_path(self, anno_path: str) -> str:
        return anno_path.replace(self.anno_postfix, self.img_postfix).replace('bboxes', self.split)
