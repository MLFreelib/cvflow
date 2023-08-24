import sys

sys.path.append('../')

from typing import List
from common.utils import *
from components.handler_component import Codes
from components.muxer_component import SourceMuxer
from components.outer_component import DisplayComponent
from components.painter_component import Tiler
from components.reader_component import ReaderBase, ImageReader, VideoReader, CamReader
from pipeline import Pipeline

from common.utils import argparser as ap

def get_muxer(readers: List[ReaderBase]) -> SourceMuxer:
    muxer = SourceMuxer('muxer', max_batch_size=1)
    for reader in readers:
        muxer.add_source(reader)
    return muxer


def get_tiler(name: str, tiler_size: tuple, frame_size: tuple = (640, 1280)) -> Tiler:
    tiler = Tiler(name=name, tiler_size=tiler_size)
    tiler.set_size(frame_size)
    return tiler


def get_usb_cam(path: str, name: str) -> CamReader:
    return CamReader(path, name)


def get_videofile_reader(path: str, name: str) -> VideoReader:
    return VideoReader(path, name)


def get_img_reader(path: str, name: str) -> ImageReader:
    return ImageReader(path, name)


if __name__ == '__main__':
    args = vars(ap.parse_args())
    pipeline = Pipeline()

    readers = []

    usb_srcs = get_cam_srcs()
    for usb_src in usb_srcs:
        readers.append(get_usb_cam(usb_src, usb_src))

    file_srcs = get_video_file_srcs()
    for file_src in file_srcs:
        readers.append(get_videofile_reader(file_src, os.path.basename(file_src)))

    img_srcs = get_img_srcs()
    for img_src in img_srcs:
        readers.append(get_img_reader(img_src, os.path.basename(img_src)))

    muxer = get_muxer(readers)

    codes = Codes('codes')

    tiler = get_tiler('tiler', tiler_size=get_tsize(), frame_size=get_fsize())
    outer = DisplayComponent('file')

    pipeline.set_device(args['device'])
    pipeline.add_all([muxer, codes, tiler, outer])
    pipeline.compile()
    pipeline.run()
    pipeline.close()
