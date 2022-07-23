import unittest

import torch

from Meta import MetaLabel, MetaBBox, MetaMask, MetaFrame, MetaBatch, MetaDepth


class TestMetaLabel(unittest.TestCase):

    def setUp(self):
        self.labels = [f'label_{i}' for i in range(10)]
        self.confs = [i / 10 for i in range(10)]
        self.meta_label = MetaLabel(self.labels, self.confs)

    def test_get_confidence(self):
        self.assertListEqual(self.confs, self.meta_label.get_confidence())

    def test_set_object_id_ValueError_exception(self):
        self.assertRaises(ValueError, self.meta_label.set_object_id, [1 for _ in range(5)])

    def test_get_object_id(self):
        ids = [1 for _ in range(10)]
        self.meta_label.set_object_id(ids)
        self.assertListEqual(ids, self.meta_label.get_object_ids())

    def test_get_labels(self):
        self.assertListEqual(self.labels, self.meta_label.get_labels())


class TestMetaBBox(unittest.TestCase):

    def setUp(self):
        self.points = torch.tensor([[0, 0, 10, 10], [12, 12, 45, 45]])
        self.labels = [f'label_{i}' for i in range(2)]
        self.confs = [i / 10 for i in range(2)]
        self.meta_label = MetaLabel(self.labels, self.confs)

    def test_points_TypeError_exception_init_points(self):
        try:
            MetaBBox(self.points.detach().numpy(), self.meta_label)
            self.assertTrue(False)
        except TypeError:
            self.assertTrue(True)

    def test_points_ValueError_exception_init_points_points_shape(self):
        try:
            MetaBBox(self.points[0], self.meta_label)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_points_ValueError_exception_init_points_bbox_shape(self):
        try:
            MetaBBox(self.points[:, :3], self.meta_label)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_points_ValueError_exception_init_label_count(self):
        meta_label = MetaLabel(self.labels[:-1], self.confs[:-1])
        try:
            MetaBBox(self.points, meta_label)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_get_bbox(self):
        meta_bbox = MetaBBox(self.points, self.meta_label)
        self.assertEqual(self.points.detach().tolist(), meta_bbox.get_bbox().detach().tolist())

    def test_get_label_info(self):
        meta_bbox = MetaBBox(self.points, self.meta_label)
        self.assertEqual(self.meta_label, meta_bbox.get_label_info())


class TestMetaMask(unittest.TestCase):

    def setUp(self):
        self.masks = torch.ones((1, 25, 240, 240))
        self.labels = [f'label_{i}' for i in range(25)]
        self.confs = [i / 10 for i in range(25)]
        self.meta_label = MetaLabel(self.labels, self.confs)

    def test_points_ValueError_exception_init_masks(self):
        try:
            MetaMask(self.masks[0], self.meta_label)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_points_ValueError_exception_init_label_count(self):
        meta_label = MetaLabel(self.labels[:-1], self.confs[:-1])
        try:
            MetaMask(self.masks, meta_label)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_get_mask(self) -> torch.tensor:
        meta_mask = MetaMask(self.masks, self.meta_label)
        self.assertEqual(self.masks.detach().tolist(), meta_mask.get_mask().detach().tolist())

    def test_get_label_info(self):
        meta_mask = MetaMask(self.masks, self.meta_label)
        self.assertEqual(self.labels, meta_mask.get_label_info().get_labels())


class TestMetaDepth(unittest.TestCase):

    def setUp(self):
        self.depth = torch.ones((1, 240, 240))

    def test_points_ValueError_exception_init_masks(self):
        try:
            MetaDepth(self.depth[0])
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_get_depth(self) -> torch.tensor:
        meta_mask = MetaDepth(self.depth)
        self.assertEqual(self.depth.detach().tolist(), meta_mask.get_depth().detach().tolist())


class TestMetaFrame(unittest.TestCase):

    def setUp(self):
        self.masks = torch.ones((1, 25, 240, 240))
        self.depth = torch.ones((1, 240, 240))
        self.labels = [f'label_{i}' for i in range(25)]
        self.confs = [i / 10 for i in range(25)]
        self.meta_label = MetaLabel(self.labels, self.confs)
        self.points = torch.tensor([[0, 0, 10, 10] for _ in range(25)])

        self.frame = torch.ones((3, 240, 240))

    def test_get_src_name(self):
        src_name = 'test_src'
        meta_frame = MetaFrame(src_name, self.frame)
        self.assertEqual(src_name, meta_frame.get_src_name())

    def test_get_frame(self):
        meta_frame = MetaFrame('test_src', self.frame)
        self.assertListEqual(self.frame.detach().tolist(), meta_frame.get_frame().detach().tolist())

    def test_get_meta_info(self):
        meta_frame = MetaFrame('test_src', self.frame)
        meta_frame.add_meta(meta_name='test_meta', value='test_value')
        self.assertEqual('test_value', meta_frame.get_meta_info('test_meta'))

    def test_get_labels_info(self):
        meta_frame = MetaFrame('test_src', self.frame)
        meta_label = MetaLabel(self.labels, self.confs)
        meta_frame.set_label_info(meta_label)
        self.assertListEqual(self.labels, meta_frame.get_labels_info().get_labels())
        self.assertListEqual(self.confs, meta_frame.get_labels_info().get_confidence())

    def test_get_mask_info(self):
        meta_frame = MetaFrame('test_src', self.frame)
        meta_label = MetaLabel(self.labels, self.confs)
        meta_mask = MetaMask(self.masks, meta_label)
        meta_frame.set_mask_info(meta_mask)
        self.assertListEqual(self.masks.detach().tolist(), meta_frame.get_mask_info().get_mask().detach().tolist())

    def test_get_bbox_info(self):
        meta_frame = MetaFrame('test_src', self.frame)
        meta_label = MetaLabel(self.labels, self.confs)
        meta_bbox = MetaBBox(self.points, meta_label)
        meta_frame.set_bbox_info(meta_bbox)
        self.assertListEqual(self.points.detach().tolist(), meta_frame.get_bbox_info().get_bbox().detach().tolist())

    def test_get_depth_info(self):
        meta_frame = MetaFrame('test_src', self.frame)
        meta_depth = MetaDepth(self.depth)
        meta_frame.set_depth_info(meta_depth)
        self.assertListEqual(self.depth.detach().tolist(), meta_frame.get_depth_info().get_depth().detach().tolist())

    def test_set_bbox_info_TypeError_exception(self):
        meta_frame = MetaFrame('test_src', self.frame)
        self.assertRaises(TypeError, meta_frame.set_bbox_info, 'mock')

    def test_set_mask_info_TypeError_exception(self):
        meta_frame = MetaFrame('test_src', self.frame)
        self.assertRaises(TypeError, meta_frame.set_mask_info, 'mock')

    def test_set_label_info_TypeError_exception(self):
        meta_frame = MetaFrame('test_src', self.frame)
        self.assertRaises(TypeError, meta_frame.set_label_info, 'mock')

    def test_set_frame_TypeError_exception(self):
        meta_frame = MetaFrame('test_src', self.frame)
        self.assertRaises(TypeError, meta_frame.set_frame, self.frame.detach().tolist())

    def test_set_frame_ValueError_exception_shape(self):
        meta_frame = MetaFrame('test_src', self.frame)
        self.assertRaises(ValueError, meta_frame.set_frame, self.frame[None, :, :, :])

    def test_set_frame_ValueError_exception_channels(self):
        meta_frame = MetaFrame('test_src', self.frame)
        self.assertRaises(ValueError, meta_frame.set_frame, self.frame[[0]])


class TestMetaBatch(unittest.TestCase):

    def setUp(self):
        self.masks = torch.ones((1, 25, 240, 240))
        self.labels = [f'label_{i}' for i in range(25)]
        self.confs = [i / 10 for i in range(25)]
        self.meta_label = MetaLabel(self.labels, self.confs)
        self.points = torch.tensor([[0, 0, 10, 10] for _ in range(25)])

        self.frame = torch.ones((3, 240, 240))

    def test_add_meta_frame_TypeError_exception(self):
        meta_batch = MetaBatch('test')
        self.assertRaises(TypeError, meta_batch.add_meta_frame, 'mock')

    def test_add_frames_TypeError_exception(self):
        meta_batch = MetaBatch('test')
        self.assertRaises(TypeError, meta_batch.add_frames, 'mock_name', 'mock_frames')

    def test_add_frames_shape_check_one_frame(self):
        meta_batch = MetaBatch('test')
        src_name = 'test_src'
        meta_batch.add_frames(src_name, self.frame)
        self.assertListEqual(list(self.frame[None, :, :, :].shape),
                             list(meta_batch.get_frames_by_src_name(src_name).shape))

    def test_get_frames_all(self):
        meta_batch = MetaBatch('test')
        test_srcs = [f'test_src_{i}' for i in range(3)]
        for src in test_srcs:
            meta_batch.add_frames(src, frames=self.frame.detach().clone())
        returned_frames = meta_batch.get_frames_all().values()
        returned_frames_shape = torch.cat(list(returned_frames), dim=0).detach().shape
        true_all_frames_shape = torch.cat([self.frame[None, :, :, :] for _ in range(len(test_srcs))],
                                          dim=0).detach().shape
        self.assertListEqual(list(true_all_frames_shape), list(returned_frames_shape))

    def test_get_meta_frames_by_src_name(self):
        meta_batch = MetaBatch('test')
        test_src = 'test_src1'
        meta_batch.add_meta_frame(MetaFrame(test_src, self.frame))
        true_frame = self.frame[None, :, :, :].detach().numpy()
        returned_frame = meta_batch.get_meta_frames_by_src_name(src_name=test_src)[0].get_frame().detach().numpy()
        self.assertTrue(not (true_frame - returned_frame).any())

    def test_get_meta_frames_all(self):
        meta_batch = MetaBatch('test')
        test_srcs = [f'test_src_{i}' for i in range(3)]
        for src in test_srcs:
            meta_batch.add_meta_frame(MetaFrame(src, self.frame))
        returned_meta_frames = torch.cat(
            [meta_frame[0].get_frame()[None, :, :, :] for meta_frame in meta_batch.get_meta_frames_all().values()],
            dim=0).detach().numpy()
        true_meta_frames = torch.cat([self.frame[None, :, :, :] for _ in range(len(test_srcs))],
                                     dim=0).detach().numpy()
        self.assertTrue(not (true_meta_frames - returned_meta_frames).any())

    def test_set_source_names_TypeError_exception(self):
        meta_batch = MetaBatch('test')
        self.assertRaises(TypeError, meta_batch.set_source_names, 'mock')

    def get_source_names(self):
        meta_batch = MetaBatch('test')
        test_srcs = [f'test_src_{i}' for i in range(3)]
        meta_batch.set_source_names(test_srcs)
        self.assertEqual(test_srcs, meta_batch.get_source_names())


if __name__ == '__main__':
    unittest.main()
