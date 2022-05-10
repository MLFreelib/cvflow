import unittest

import cv2
import numpy as np
import torch
import torchvision
from torchvision import models

from Meta import MetaBatch, MetaFrame, MetaBBox, MetaMask, MetaLabel
from components.model_component import ModelDetection, ModelSegmentation, ModelClassification


def _to_meta_frame(frame: np.array, src_name: str, device: str = 'cpu') -> MetaFrame:
    r""" Creates MetaFrame from a frame. """
    frame = torch.tensor(frame, device=device)
    frame = frame.permute(2, 0, 1)
    meta_frame = MetaFrame(source_name=src_name, frame=frame)
    return meta_frame


class ModelDetectionTest(unittest.TestCase):

    def setUp(self):
        self.resnet50 = models.detection.retinanet_resnet50_fpn(pretrained=True)
        self.resnet50.eval()
        self.model = ModelDetection('detection_test', self.resnet50)
        self.model.set_device('cpu')
        self.model.set_confidence(.0)
        self.model.set_labels([f'{i}' for i in range(91)])
        self.model.set_transforms([torchvision.transforms.Resize((480, 640))])

        meta_batch = MetaBatch('test_batch')
        for i in range(2):
            src_name = f'test_frame{i}'
            self.model.add_source(src_name)
            frame = cv2.imread(filename='tests/test_data/zebra.jpg')
            meta_frame = _to_meta_frame(frame=frame, src_name=src_name)
            meta_batch.add_meta_frame(meta_frame)
            meta_batch.add_frames(src_name, meta_frame.get_frame())

        self.meta_batch = self.model.do(meta_batch)

    def test_do_bbox_exists(self):
        meta_bboxes = self.meta_batch.get_meta_frames_by_src_name('test_frame0')[0].get_bbox_info()
        self.assertIsNotNone(meta_bboxes)

    def test_do_bbox_type_correct(self):
        meta_bboxes = self.meta_batch.get_meta_frames_by_src_name('test_frame0')[0].get_bbox_info()
        self.assertEqual(MetaBBox, type(meta_bboxes))

    def tearDown(self):
        del self.model
        del self.resnet50


class ModelSegmentationTest(unittest.TestCase):

    def setUp(self):
        self.resnet50 = models.segmentation.fcn_resnet50(pretrained=True)
        self.resnet50.eval()
        self.model = ModelSegmentation('segmentation_test', self.resnet50)
        self.model.set_device('cpu')
        self.model.set_confidence(.0)
        self.model.set_labels([f'{i}' for i in range(21)])

        meta_batch = MetaBatch('test_batch')
        for i in range(2):
            src_name = f'test_frame{i}'
            self.model.add_source(src_name)
            frame = cv2.imread(filename='tests/test_data/mouse.jpeg')
            meta_frame = _to_meta_frame(frame=frame, src_name=src_name)
            meta_batch.add_meta_frame(meta_frame)
            meta_batch.add_frames(src_name, meta_frame.get_frame())

        self.meta_batch = self.model.do(meta_batch)

    def test_do_mask_exists(self):
        meta_mask = self.meta_batch.get_meta_frames_by_src_name('test_frame0')[0].get_mask_info()
        self.assertIsNotNone(meta_mask)

    def test_do_mask_type_correct(self):
        meta_mask = self.meta_batch.get_meta_frames_by_src_name('test_frame0')[0].get_mask_info()
        self.assertEqual(MetaMask, type(meta_mask))

    def tearDown(self):
        del self.model
        del self.resnet50


class ModelClassificationTest(unittest.TestCase):

    def setUp(self):
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.eval()
        self.model = ModelClassification('classification_test', self.resnet50)
        self.model.set_device('cpu')
        self.model.set_confidence(.0)
        self.model.set_labels([f'{i}' for i in range(1000)])

        meta_batch = MetaBatch('test_batch')
        for i in range(2):
            src_name = f'test_frame{i}'
            self.model.add_source(src_name)
            frame = cv2.imread(filename='tests/test_data/mouse.jpeg')
            meta_frame = _to_meta_frame(frame=frame, src_name=src_name)
            meta_batch.add_meta_frame(meta_frame)
            meta_batch.add_frames(src_name, meta_frame.get_frame())

        self.meta_batch = self.model.do(meta_batch)

    def test_do_mask_exists(self):
        meta_label = self.meta_batch.get_meta_frames_by_src_name('test_frame0')[0].get_labels_info()
        self.assertIsNotNone(meta_label)

    def test_do_mask_type_correct(self):
        meta_label = self.meta_batch.get_meta_frames_by_src_name('test_frame0')[0].get_labels_info()
        self.assertEqual(MetaLabel, type(meta_label))

    def tearDown(self):
        del self.model
        del self.resnet50


if __name__ == '__main__':
    unittest.main()