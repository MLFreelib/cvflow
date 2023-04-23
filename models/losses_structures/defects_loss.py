import torch
from torch import nn
from torch.nn import TripletMarginLoss

from Meta import MetaBatch, MetaName
from models.losses_structures.loss_base import LossBase
from models.utils import cxcy_to_xy, find_jaccard_overlap, xy_to_cxcy, cxcy_to_gcxgcy


class MultiBoxLoss(LossBase):
    """
    The MultiBox loss, a loss function for object detection.

    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1., beta=1., emb_dim=512):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        self.beta = beta
        self.emb_dim = emb_dim

        self.smooth_l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.triplet_loss = TripletMarginLoss()

    def forward(self, predicted_locs, predicted_triplets, boxes):
        """
        Forward propagation.
        predicted_locs, predicted_triplets, boxes
        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param predicted_triplets: predicted embeddings for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, emb_dim)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """

        batch_size = len(predicted_locs)
        device = boxes[0].get_device()
        device = 'cpu' if device == -1 else device

        true_locs = list()  # (N, 8732, 4)

        # For each image
        for i in range(batch_size):
            overlap = find_jaccard_overlap(boxes[i], self.priors_xy.to(device))  # (n_objects, 8732)
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs.append(
                cxcy_to_gcxgcy(xy_to_cxcy(boxes[i]), self.priors_cxcy.to(device)[prior_for_each_object]))  # (8732, 4)

        # LOCALIZATION LOSS

        batch_predicted_locs = torch.concat(predicted_locs, dim=0)
        batch_true_locs = torch.concat(true_locs, dim=0)
        loc_loss = self.smooth_l1(batch_predicted_locs, batch_true_locs)  # (), scalar

        # Triplet Loss for embeddings
        batch_predicted_triplets = torch.concat(predicted_triplets, dim=0)

        triplet_loss = self.triplet_loss(batch_predicted_triplets[:, 0, :self.emb_dim],
                                         batch_predicted_triplets[:, 1, :self.emb_dim],
                                         batch_predicted_triplets[:, 2, :self.emb_dim])

        return {'boxes': loc_loss, 'triplets': triplet_loss}, loc_loss + triplet_loss

    def __call__(self, predicted_values, true_values: MetaBatch, **kwargs):
        predicted_locs = [value['boxes'] for value in predicted_values]
        device = predicted_locs[0].get_device()
        device = 'cpu' if device == -1 else device
        predicted_triplets = [value['embeddings'] for value in predicted_values]
        key = list(true_values.get_meta_frames_all().keys())[0]
        boxes = [value.get_meta_info(MetaName.META_BBOX.value).get_bbox().to(device)
                 for value in true_values.get_meta_frames_by_src_name(key)]
        return self.forward(predicted_locs, predicted_triplets, boxes)
