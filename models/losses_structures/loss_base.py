from typing import Dict, Tuple, Any

import torch
from torch import nn

from Meta import MetaBatch


class LossBase(nn.Module):
    r"""
        :return: Tuple[Dict[str, torch.Tensor], torch.Tensor]
        Dict[str, torch.Tensor] - {Name of loss: loss value, ... }
        torch.Tensor - full loss
    """

    def __call__(self, predict: Any, batch_info: MetaBatch, *args, **kwargs) -> Tuple[
        Dict[str, torch.Tensor], torch.Tensor]:
        return {}, torch.tensor(0)
