import os.path
import unittest

from Meta import MetaBatch
from components.muxer_component import SourceMuxer
from components.reader_component import VideoReader


class SourceMuxerTest(unittest.TestCase):

    def setUp(self):
        self.reader = VideoReader('tests/test_data/Lion Animal.mp4', 'reader')

    def test_add_source_TypeError_exception(self):
        muxer = SourceMuxer('muxer')
        self.assertRaises(TypeError, muxer.add_source, 'mock')

    def test_do_type_check(self):
        muxer = SourceMuxer('muxer')
        muxer.add_source(self.reader)
        muxer.start()
        self.assertEqual(MetaBatch, type(muxer.do(MetaBatch('test_batch'))))
        muxer.stop()
