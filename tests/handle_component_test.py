import unittest

import torch
import sys

sys.path.append('../')

from cvflow.Meta import MetaFrame, MetaLabel, MetaBatch, MetaBBox, MetaMask
from cvflow.components.handler_component import Filter, Counter


class FilterTest(unittest.TestCase):

    def setUp(self):
        self.labels = [f'test_label_{i}' for i in range(10)]
        self.confs = torch.tensor([i / 10 for i in range(10)])
        self.ids = [i for i in range(10)]
        self.points = torch.tensor([[i, i, i, i] for i in range(10)])
        self.masks = torch.ones((1, len(self.labels), 20, 20), dtype=torch.uint8, device='cpu')
        self.filter = Filter('test_filter', self.labels[:5])
        self.frame = torch.ones((3, 30, 30), dtype=torch.uint8, device='cpu')

    def test_start_TypeError_exception(self):
        test_filter = Filter('test_filter', 'mock')
        self.assertRaises(TypeError, test_filter.start)

    def test_do_for_labels_in_classification(self):
        returned_meta_label = self.__get_meta_label()
        correct_labels = list(set(self.labels[:5]) & set(self.labels[3:8]))
        correct_labels = sorted(correct_labels)
        self.assertListEqual(returned_meta_label.get_labels(), correct_labels)

    def test_do_for_confs_in_classification(self):
        returned_meta_label = self.__get_meta_label()
        correct_confs = list(set(self.confs[:5].detach().cpu().tolist()) & set(self.confs[3:8].detach().cpu().tolist()))
        correct_confs = sorted(correct_confs)
        returned_confs = returned_meta_label.get_confidence()[0].detach().cpu().tolist()
        self.assertListEqual(returned_confs, correct_confs)

    def test_do_for_labels_in_detection(self):
        returned_meta_bbox = self.__get_meta_bbox()
        returned_meta_label = returned_meta_bbox.get_label_info()
        correct_labels = list(set(self.labels[:5]) & set(self.labels[3:8]))
        correct_labels = sorted(correct_labels)
        self.assertListEqual(returned_meta_label.get_labels(), correct_labels)

    def test_do_for_confs_in_detection(self):
        returned_meta_bbox = self.__get_meta_bbox()
        returned_meta_label = returned_meta_bbox.get_label_info()
        correct_confs = list(set(self.confs[:5].detach().cpu().tolist()) & set(self.confs[3:8].detach().cpu().tolist()))
        correct_confs = sorted(correct_confs)
        returned_confs = returned_meta_label.get_confidence().detach().cpu().tolist()
        self.assertListEqual(returned_confs, correct_confs)

    def test_do_for_ids_in_detection(self):
        returned_meta_bbox = self.__get_meta_bbox()
        returned_meta_label = returned_meta_bbox.get_label_info()
        correct_ids = list(set(self.ids[:5]) & set(self.ids[3:8]))
        correct_ids = sorted(correct_ids)
        self.assertListEqual(returned_meta_label.get_object_ids(), correct_ids)

    def test_do_for_masks_in_segmentation(self):
        returned_meta_mask = self.__get_meta_mask()
        correct_masks = self.masks.clone()[:, 3: 8]
        correct_masks[:, 3: 5] = 0
        correct_masks = correct_masks.detach().cpu().tolist()
        returned_masks = returned_meta_mask.get_mask().clone()
        returned_masks = returned_masks.detach().cpu().tolist()
        self.assertListEqual(returned_masks, correct_masks)

    def __get_meta_mask(self):
        test_src_name = 'test_meta_frame'
        meta_frame = MetaFrame(test_src_name, self.frame)
        meta_label = MetaLabel(self.labels[3: 8], self.confs[3: 8])
        meta_label.set_object_id(self.ids[3: 8])
        meta_mask = MetaMask(self.masks[:, 3: 8], meta_label)
        meta_frame.set_mask_info(meta_mask)
        meta_batch = MetaBatch(test_src_name)
        meta_batch.add_meta_frame(meta_frame)
        meta_batch.set_source_names([test_src_name])
        returned_meta_batch = self.filter.do(meta_batch)
        returned_meta_frame = returned_meta_batch.get_meta_frames_by_src_name(test_src_name)[0]
        returned_meta_mask = returned_meta_frame.get_mask_info()
        return returned_meta_mask

    def __get_meta_bbox(self):
        test_src_name = 'test_meta_frame'
        meta_frame = MetaFrame(test_src_name, self.frame)
        meta_label = MetaLabel(self.labels[3: 8], self.confs[3: 8])
        meta_label.set_object_id(self.ids[3: 8])
        meta_bbox = MetaBBox(self.points[3: 8], meta_label)
        meta_frame.set_bbox_info(meta_bbox)
        meta_batch = MetaBatch(test_src_name)
        meta_batch.add_meta_frame(meta_frame)
        meta_batch.set_source_names([test_src_name])
        returned_meta_batch = self.filter.do(meta_batch)
        returned_meta_frame = returned_meta_batch.get_meta_frames_by_src_name(test_src_name)[0]
        returned_meta_bbox = returned_meta_frame.get_bbox_info()
        return returned_meta_bbox

    def __get_meta_label(self):
        test_src_name = 'test_meta_frame'
        meta_frame = MetaFrame(test_src_name, self.frame)
        meta_label = MetaLabel(self.labels[3:8], self.confs[3: 8])
        meta_frame.set_label_info(meta_label)
        meta_batch = MetaBatch('test_meta_batch')
        meta_batch.add_meta_frame(meta_frame)
        meta_batch.set_source_names([test_src_name])
        returned_meta_batch = self.filter.do(meta_batch)
        returned_meta_frame = returned_meta_batch.get_meta_frames_by_src_name(test_src_name)[0]
        returned_meta_label = returned_meta_frame.get_labels_info()
        return returned_meta_label

<<<<<<< HEAD

class CounterTest(unittest.TestCase):

    def setUp(self):
        self.frame = torch.zeros([3, 100, 100])
        self.line = [0, 0, 50, 50]
        self.bbox_in = torch.tensor([0.3, 0.3, 0.4, 0.4])
        self.bbox_out = torch.tensor([0.0, 0.9, 0.1, 1])
        self.labels = [f'test_label_{i}' for i in range(3)]
        self.confs = [i / 10 for i in range(3)]
        self.ids = [i for i in range(3)]

    def test_do_for_one_in(self):
        src_name = 'test_src'
        counter = Counter('counter', self.line)
        returned_meta_batch = None
        label_num = 0
        for _ in range(3):
            meta_batch = self.__get_meta_batch(src_name, self.bbox_in, label_id=label_num)
            returned_meta_batch = counter.do(meta_batch)

        label_count = returned_meta_batch.get_meta_frames_by_src_name(src_name)[0].get_meta_info('counter')

        self.assertEqual(1, label_count['labels'][self.labels[label_num]])
        self.assertEqual(1, label_count['ids'][self.ids[label_num]])

    def test_do_zero_in(self):
        src_name = 'test_src'
        counter = Counter('counter', self.line)
        returned_meta_batch = None
        label_num = 0
        for _ in range(3):
            meta_batch = self.__get_meta_batch(src_name, self.bbox_out, label_id=label_num)
            returned_meta_batch = counter.do(meta_batch)

        label_count = returned_meta_batch.get_meta_frames_by_src_name(src_name)[0].get_meta_info('counter')
        self.assertEqual(0, len(label_count['labels'].keys()))
        self.assertEqual(0, len(label_count['ids'].keys()))

    def test_do_one_in_one_out(self):
        src_name = 'test_src'
        counter = Counter('counter', self.line)
        returned_meta_batch = None
        for _ in range(2):
            meta_batch = self.__get_meta_batch(src_name, self.bbox_in, label_id=0, ids=0)
            returned_meta_batch = counter.do(meta_batch)

        for _ in range(2):
            meta_batch = self.__get_meta_batch(src_name, self.bbox_out, label_id=0, ids=1)
            returned_meta_batch = counter.do(meta_batch)

        label_count = returned_meta_batch.get_meta_frames_by_src_name(src_name)[0].get_meta_info('counter')
        self.assertEqual(1, label_count['labels'][self.labels[0]])
        self.assertEqual(1, label_count['ids'][self.ids[0]])

    def test_do_one_in_out_in(self):
        src_name = 'test_src'
        counter = Counter('counter', self.line)
        returned_meta_batch = None
        for _ in range(2):
            meta_batch = self.__get_meta_batch(src_name, self.bbox_in, label_id=0, ids=0)
            returned_meta_batch = counter.do(meta_batch)

        for _ in range(2):
            meta_batch = self.__get_meta_batch(src_name, self.bbox_out, label_id=0, ids=0)
            returned_meta_batch = counter.do(meta_batch)

        for _ in range(2):
            meta_batch = self.__get_meta_batch(src_name, self.bbox_in, label_id=0, ids=0)
            returned_meta_batch = counter.do(meta_batch)

        label_count = returned_meta_batch.get_meta_frames_by_src_name(src_name)[0].get_meta_info('counter')
        self.assertEqual(2, label_count['labels'][self.labels[0]])
        self.assertEqual(2, label_count['ids'][self.ids[0]])

    def test_do_one_in_out_in_with_different_label(self):
        src_name = 'test_src'
        counter = Counter('counter', self.line)
        returned_meta_batch = None
        for _ in range(2):
            meta_batch = self.__get_meta_batch(src_name, self.bbox_in, label_id=0, ids=0)
            returned_meta_batch = counter.do(meta_batch)

        for _ in range(2):
            meta_batch = self.__get_meta_batch(src_name, self.bbox_out, label_id=0, ids=0)
            returned_meta_batch = counter.do(meta_batch)

        for _ in range(2):
            meta_batch = self.__get_meta_batch(src_name, self.bbox_in, label_id=1, ids=0)
            returned_meta_batch = counter.do(meta_batch)

        label_count = returned_meta_batch.get_meta_frames_by_src_name(src_name)[0].get_meta_info('counter')
        self.assertEqual(2, len(label_count['labels'].keys()))
        self.assertEqual(2, label_count['ids'][self.ids[0]])

    def test_do_one_in_out_in_with_different_id(self):
        src_name = 'test_src'
        counter = Counter('counter', self.line)
        returned_meta_batch = None
        for _ in range(2):
            meta_batch = self.__get_meta_batch(src_name, self.bbox_in, label_id=0, ids=0)
            returned_meta_batch = counter.do(meta_batch)

        for _ in range(2):
            meta_batch = self.__get_meta_batch(src_name, self.bbox_out, label_id=0, ids=0)
            returned_meta_batch = counter.do(meta_batch)

        for _ in range(2):
            meta_batch = self.__get_meta_batch(src_name, self.bbox_in, label_id=0, ids=1)
            returned_meta_batch = counter.do(meta_batch)

        label_count = returned_meta_batch.get_meta_frames_by_src_name(src_name)[0].get_meta_info('counter')
        self.assertEqual(2, label_count['labels'][self.labels[0]])
        self.assertEqual(2, len(label_count['ids'].keys()))

    def __get_meta_batch(self, src_name, bbox: torch.Tensor, label_id: int = 0, ids: int = 0):
        meta_label = MetaLabel([self.labels[label_id]], [self.confs[label_id]])
        meta_label.set_object_id([self.ids[ids]])
        meta_bbox = MetaBBox(torch.unsqueeze(bbox, dim=0), meta_label)
        meta_frame = MetaFrame(src_name, self.frame)
        meta_frame.set_bbox_info(meta_bbox)
        meta_batch = MetaBatch('mock')
        meta_batch.add_meta_frame(meta_frame)
        meta_batch.set_source_names([src_name])
        return meta_batch

if __name__ == '__main__':
    unittest.main()
