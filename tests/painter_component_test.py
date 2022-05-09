import unittest

import torch

from Meta import MetaBatch, MetaFrame, MetaLabel, MetaBBox, MetaMask
from components.painter_component import Tiler, BBoxPainter, LabelPainter, MaskPainter


class TilerTest(unittest.TestCase):

    def setUp(self):
        self.tiler = Tiler(name='tiler', tiler_size=(2, 2))
        self.size = (40, 60)
        self.tiler.set_size(self.size)

    def test_do_check_frame_size(self):
        meta_batch = self.__get_meta_frame(4)
        meta_frames_tiler = meta_batch.get_meta_frames_by_src_name('tiler')[0]
        self.assertListEqual(list(self.size), list(meta_frames_tiler.get_frame().shape[-2:]))

    def test_do_check_if_not_enough_frames(self):
        meta_batch = self.__get_meta_frame(3)
        meta_frames_tiler = meta_batch.get_meta_frames_by_src_name('tiler')[0]
        self.assertListEqual(list(self.size), list(meta_frames_tiler.get_frame().shape[-2:]))

    def __get_meta_frame(self, frames_number: int):
        meta_batch = MetaBatch('mock')
        source_names = [f'test_source_{i}' for i in range(frames_number)]
        for i in range(frames_number):
            frame = torch.ones((3, 20, 20))
            meta_batch.add_meta_frame(MetaFrame(source_name=source_names[i], frame=frame))
        meta_batch.set_source_names(source_names)
        meta_batch = self.tiler.do(meta_batch)
        return meta_batch


class BBoxPainterTest(unittest.TestCase):

    def setUp(self):
        bboxes = torch.tensor([[.1, .2, .2, .5],
                               [.2, .4, .5, .6]])
        labels_name = [f'label_{i}' for i in range(bboxes.shape[0])]
        labels_conf = [0.1, 0.8]
        meta_label = MetaLabel(labels=labels_name, confidence=labels_conf)
        self.frame = torch.ones((3, 40, 60), dtype=torch.uint8)
        self.meta_frame = MetaFrame('test_src', self.frame)
        self.meta_frame.set_bbox_info(MetaBBox(bboxes, meta_label))
        self.meta_batch = MetaBatch('mock')
        self.meta_batch.add_meta_frame(self.meta_frame)
        self.meta_batch.set_source_names(['test_src'])

    def test_do_check_draw_bboxes(self):
        bbox_painter = BBoxPainter('painter', 'fonts/OpenSans-VariableFont_wdth,wght.ttf')
        self.meta_batch = bbox_painter.do(self.meta_batch)
        frames_dif = self.meta_batch.get_meta_frames_by_src_name('test_src')[0].get_frame() - self.frame
        self.assertTrue(torch.sum(frames_dif) != 0)


class LabelPainterTest(unittest.TestCase):

    def setUp(self):
        labels_name = [f'label_{i}' for i in range(2)]
        labels_conf = torch.tensor([[0.1, 0.8]])
        meta_label = MetaLabel(labels=labels_name, confidence=labels_conf)
        self.frame = torch.ones((3, 40, 60), dtype=torch.uint8)
        self.meta_frame = MetaFrame('test_src', self.frame)
        self.meta_batch = MetaBatch('mock')
        self.meta_frame.set_label_info(meta_label)
        self.meta_batch.add_meta_frame(self.meta_frame)
        self.meta_batch.set_source_names(['test_src'])

    def test_do_check_draw_labels(self):
        label_painter = LabelPainter('painter')
        self.meta_batch = label_painter.do(self.meta_batch)
        frames_dif = self.meta_batch.get_meta_frames_by_src_name('test_src')[0].get_frame() - self.frame
        self.assertTrue(torch.sum(frames_dif) != 0)


class MaskPainterTest(unittest.TestCase):

    def setUp(self):
        labels_name = [f'label_{i}' for i in range(2)]
        labels_conf = torch.tensor([[0.1, 0.8]])
        meta_label = MetaLabel(labels=labels_name, confidence=labels_conf)
        self.frame = torch.ones((3, 240, 360), dtype=torch.uint8)
        mask = torch.randint(0, 2, size=(1, 2, 240, 360), dtype=torch.bool)
        self.meta_frame = MetaFrame('test_src', self.frame)
        self.meta_batch = MetaBatch('mock')
        self.meta_frame.set_mask_info(MetaMask(mask, meta_label))
        self.meta_batch.add_meta_frame(self.meta_frame)
        self.meta_batch.set_source_names(['test_src'])

    def test_do_check_draw_masks(self):
        mask_painter = MaskPainter('painter')
        self.meta_batch = mask_painter.do(self.meta_batch)
        frames_dif = self.meta_batch.get_meta_frames_by_src_name('test_src')[0].get_frame() - self.frame
        self.assertTrue(torch.sum(frames_dif) != 0)


if __name__ == '__main__':
    unittest.main()

