from typing import Union

from torch import nn

from models.blocks import Block
from models.models import ModelBuilder


class DefectsModel(ModelBuilder):

    def __init__(self, input_block: Union[Block, nn.Module], backbone: Union[Block, nn.Module],
                 output_block: Union[Block, nn.Module]):
        super().__init__(input_block, backbone, output_block)
        self._in
