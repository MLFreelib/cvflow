import os
import unittest

import numpy as np

from components.reader_component import VideoReader


class VideoReaderTest(unittest.TestCase):
    def test_read_type_check(self):
        reader = VideoReader(os.path.join(os.path.dirname(__file__), 'test_data/Lion Animal.mp4'), name='reader')
        reader.run()
        self.assertEqual(np.ndarray, type(reader.read()))
        reader.stop()

if __name__ == '__main__':
    unittest.main()
