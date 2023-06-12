import torch.nn.functional as F
import torch.nn as nn

from Meta import *
from models.losses_structures.loss_base import LossBase

class StereoLoss(LossBase):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, weights=[0.5, 0.5, 0.7, 1.0]):
        super().__init__()

        self.weights = weights

    def forward(self, disp_ests, disp_gt):
        all_losses = []
        mask = (disp_gt < 192) & (disp_gt > 0)
        # print(disp_ests, disp_gt)
        for disp_est, weight in zip(disp_ests, self.weights):
            all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], reduction='mean'))
        return {'model_loss': sum(all_losses)}, sum(all_losses)

    def __call__(self, predicted_values, true_values, **kwargs):
        # device = predicted_values[0].get_device()
        # device = 'cpu' if device == -1 else device
        # key = list(true_values.get_meta_frames_all().keys())[0]
        # depth = [value.get_meta_info(MetaName.META_DEPTH.value).get_depth().to(device)
        #          for value in true_values.get_meta_frames_by_src_name(key)]
        return self.forward(predicted_values, true_values)

