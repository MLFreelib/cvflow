from math import sqrt
from typing import Union, List

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import nms, box_iou
from tqdm import tqdm

from Meta import MetaName
from models.blocks import Block
from models.defects.blocks import AuxiliaryConvolutions, PredictionConvolutions
from models.defects.dataset import DefectsModelDataset
from models.defects.utils import find_jaccard_overlap, get_embeddings, get_seq_triplets
from models.defects.vgg19 import VGGBase
from models.utils import cxcy_to_xy, gcxgcy_to_cxcy


class SSD300(Block):
    """
    The SSD300 network - encapsulates the base VGG network, auxiliary, and prediction convolutions.
    """

    def __init__(self, emb_dim: int = 1024):
        super(SSD300, self).__init__(-1, -1)
        self.is_train = False
        self.base = VGGBase()
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(emb_dim)

        # Since lower level features (conv4_3_feats) have considerably larger scales, we take the L2 norm and rescale
        # Rescale factor is initially set at 20, but is learned for each channel during back-prop
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in conv4_3_feats
        nn.init.constant_(self.rescale_factors, 20)

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()
        self.priors_xy = cxcy_to_xy(self.priors_cxcy)

        self.last_embeddings = torch.zeros(size=(8, emb_dim), device=self.device)
        self.last_labels = torch.tensor(data=list(range(len(self.last_embeddings))), device=self.device)

        self.template_embeddings = None
        self.template_labels = None
        self.classes = None
        self.euclid_mean = None
        self.euclid_max_dist = None
        self.path_to_templates = None

    def load_templates(self, path_to_templates: str, format_img: str = 'jpg'):
        self.path_to_templates = path_to_templates
        template_embeddings = list()
        template_labels = list()
        dataset = DefectsModelDataset(path_to_templates,
                                      split='templates',
                                      keep_difficult=True, anno_postfix='_anno.txt', img_postfix='.bmp')
        with tqdm(total=len(dataset)) as t, torch.no_grad():
            for sample in dataset:
                image, boxes, labels = sample['image'], sample[MetaName.META_BBOX.value].get_bbox(), sample[MetaName.META_BBOX.value].get_label_info().get_labels()
                t.update(1)

                boxes = boxes.to(self.device)
                labels = torch.tensor(labels, device=self.device)
                embeddings = self.forward_embeddings(torch.unsqueeze(image, dim=0).to(device=self.device))

                prior_for_each_object = self.find_pos_locs([boxes])
                true_embeddings, true_labels = get_embeddings(embeddings, [labels], prior_for_each_object)
                template_embeddings.append(true_embeddings)
                template_labels.append(true_labels)

        self.template_embeddings = torch.concat(template_embeddings, dim=0).to(self.device)
        self.template_labels = torch.concat(template_labels, dim=0).to(self.device)
        self.classes = {value: dataset.le.inverse_transform([value])[0]
                        for value in self.template_labels.unique().cpu().tolist()}
        print(self.classes)

    def build_clusters(self):
        classes_count = len(list(self.classes.keys()))
        euclid_mean = torch.zeros(size=(classes_count, self.template_embeddings.shape[1]), device=self.device)
        euclid_max_dist = torch.zeros(size=(classes_count,), device=self.device)
        for label in self.template_labels.unique():
            label_embeddings = self.template_embeddings[self.template_labels == label]
            euclid_mean[label] = label_embeddings.mean(dim=0)
            euclid_max_dist[label] = torch.dist(euclid_mean[label], label_embeddings).max()
        self.euclid_mean = euclid_mean
        self.euclid_max_dist = euclid_max_dist

    def build_background(self):
        template_embeddings = list()
        dataset = DefectsModelDataset(self.path_to_templates,
                                      split='templates',
                                      keep_difficult=True, anno_postfix='_anno.txt', img_postfix='.bmp')

        with tqdm(total=len(dataset)) as t, torch.no_grad():
            for image, boxes, labels, difficulties in dataset:
                t.update(1)

                boxes = boxes.to(self.device)
                embeddings = self.forward_embeddings(torch.unsqueeze(image, dim=0).to(self.device))
                predicted_locs, predicted_labels, predicted_scores, predicted_embeddings = \
                    self.forward_eval(embeddings, threshold=.5)

                b_iou = box_iou(predicted_locs[0], boxes)
                b_iou_mask = (b_iou < .5).all(dim=1)
                predicted_scores_mask = predicted_scores[0]
                predicted_scores_mask[(b_iou > .5).all(dim=1)] = 0
                topk_values, topk_indexes = torch.topk(predicted_scores_mask, k=5)
                predicted_scores_mask[~topk_indexes] = 0
                b_iou_mask = b_iou_mask & (predicted_scores_mask > .8)
                background_embeddings = predicted_embeddings[0][b_iou_mask]
                template_embeddings.append(background_embeddings)

            template_embeddings = torch.concat(template_embeddings, dim=0)
            self.template_embeddings = torch.concat(tensors=[self.template_embeddings, template_embeddings])
            self.template_labels = torch.concat(
                tensors=[self.template_labels,
                         torch.ones(size=(template_embeddings.shape[0],), device=self.device, dtype=torch.long) * len(
                             self.classes.keys())], dim=0)
            self.classes[len(self.classes.keys())] = 'background'

    def forward_embeddings(self, image):
        conv4_3_feats, conv7_feats = self.base(image)  # (N, 512, 38, 38), (N, 1024, 19, 19)

        # Rescale conv4_3 after L2 norm
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        conv4_3_feats = conv4_3_feats / norm  # (N, 512, 38, 38)
        conv4_3_feats = conv4_3_feats * self.rescale_factors  # (N, 512, 38, 38)
        # (PyTorch autobroadcasts singleton dimensions during arithmetic)

        # Run auxiliary convolutions (higher level feature map generators)
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = \
            self.aux_convs(conv7_feats)  # (N, 512, 10, 10),  (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        embeddings = self.pred_convs.forward_embeddings(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats,
                                                        conv10_2_feats, conv11_2_feats)

        return embeddings

    def forward_train(self, image, boxes, labels):
        """
            Forward propagation.

            :param image: images, a tensor of dimensions (N, 3, 300, 300)
            :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        embeddings = self.forward_embeddings(image)

        object_count = [object_labels.shape[0] for object_labels in labels]

        prior_for_each_object = self.find_pos_locs(boxes)
        true_embeddings, true_labels = get_embeddings(embeddings, [torch.tensor(label) for label in labels], prior_for_each_object)
        del embeddings
        unique_values, count_values = torch.unique(true_labels, return_counts=True)
        triplets = get_seq_triplets(true_embeddings.to(self.device), true_labels.to(self.device),
                                    self.last_embeddings.to(self.device), self.last_labels.to(self.device))

        with torch.no_grad():
            for i in range(unique_values.shape[0]):
                indexes = (true_labels == unique_values[i]).nonzero(as_tuple=True)[0]
                self.last_embeddings[unique_values[i]] = true_embeddings[indexes[0]]
        locs = self.pred_convs.loc_layer(torch.concat([triplets[:, 0, :], triplets[:, 1, :]], dim=1))

        batch_triplets = list()
        batch_locs = list()
        prev_count = 0
        for count in object_count:
            batch_triplets.append(triplets[prev_count: prev_count + count])
            batch_locs.append(locs[prev_count: prev_count + count])
            prev_count = prev_count + count

        del triplets

        return batch_locs, batch_triplets

    def get_top(self, first_embeddings: torch.Tensor, second_embeddings: torch.Tensor, top_of):
        dist_matrix = torch.cdist(first_embeddings, second_embeddings)
        topk_values, topk_indexes = torch.topk(dist_matrix, k=5, dim=1, largest=False)
        topk_labels = top_of[topk_indexes.view(-1)]
        topk_labels = topk_labels.view(-1, 5)
        topk_mode = torch.mode(topk_labels, dim=1)
        return topk_mode

    def get_score2(self, embeddings: torch.tensor) -> Union[torch.Tensor, torch.Tensor]:
        dist_matrix_templates = torch.cdist(embeddings, self.template_embeddings)
        label_distances = torch.zeros(size=(embeddings.shape[0], self.template_labels.unique().shape[0]), device=self.device)
        for label in self.template_labels.unique():
            label_min_dist = dist_matrix_templates[:, self.template_labels == label]
            label_min_dist = torch.min(label_min_dist, dim=1).values
            label_distances[:, label] = label_min_dist
        probabilities = torch.softmax(label_distances, dim=1)
        return torch.max(probabilities, dim=1)

    def get_scores(self, embeddings: torch.tensor) -> Union[torch.Tensor, torch.Tensor]:
        # dist_matrix_templates = torch.cdist(embeddings, self.template_embeddings)
        # dist_matrix_mean = torch.cdist(embeddings, self.euclid_mean)


        dist_matrix_templates = torch.cdist(embeddings, self.template_embeddings)
        topk_values, topk_indexes = torch.topk(dist_matrix_templates, k=5, dim=1, largest=False)
        topk_labels = self.template_labels[topk_indexes.view(-1)]
        topk_labels = topk_labels.view(-1, 5)
        topk_mode = torch.mode(topk_labels, dim=1)

        topk_labels_mask = topk_labels == topk_mode.values.view(-1, 1)

        mean_dist = torch.zeros_like(topk_labels_mask, device=self.device, dtype=torch.float32)
        mean_dist[topk_labels_mask] = topk_values[topk_labels_mask]
        mean_dist = torch.sum(mean_dist, dim=1) / torch.sum((mean_dist != 0), dim=1)
        dist_matrix_mean = torch.cdist(embeddings, self.euclid_mean)

        scores = torch.zeros(size=(embeddings.shape[0],), device=self.device)
        for label in topk_labels.unique():
            indexes_by_label = topk_mode.values == label
            scores[indexes_by_label] = dist_matrix_mean[indexes_by_label, label]
            scores[indexes_by_label] = 1 - (scores[indexes_by_label] / (self.euclid_max_dist[label] + 1e-10))

        return scores, topk_mode.values

    def forward_eval(self, embeddings, threshold: float = 0.9):
        predicted_labels = list()
        predicted_locs = list()
        predicted_scores = list()
        predicted_embeddings = list()

        for i in range(embeddings.shape[0]):
            scores, labels = self.get_scores(embeddings[i])
            dist_matrix = torch.cdist(embeddings[i], self.template_embeddings)
            templates_mask = (scores > threshold)
            _, best_templates = torch.min(dist_matrix, dim=1)
            best_templates = best_templates[templates_mask]
            best_template_embeddings = self.template_embeddings[best_templates]
            best_predicted_embeddings = embeddings[i][templates_mask]
            full_embeddings = torch.concat([best_predicted_embeddings, best_template_embeddings], dim=1)
            best_predicted_locs = self.pred_convs.loc_layer(full_embeddings)
            best_predicted_locs = cxcy_to_xy(gcxgcy_to_cxcy(best_predicted_locs,
                                                            self.priors_cxcy.to(self.device)[templates_mask]))

            indexes = nms(best_predicted_locs, scores[templates_mask], iou_threshold=0.5)
            predicted_labels.append(labels[indexes])
            predicted_locs.append(best_predicted_locs[indexes])
            predicted_scores.append(scores[templates_mask][indexes])
            predicted_embeddings.append(best_predicted_embeddings[indexes])

        return predicted_locs, predicted_labels, predicted_scores, predicted_embeddings

    def forward(self, images, boxes=None, labels=None, thresholds: Union[List[float], float] = 0.1):
        if self.is_train:
            return self.forward_train(images, boxes, labels)
        else:
            embeddings = self.forward_embeddings(images)
            predicted_locs, predicted_labels, predicted_scores, predicted_embeddings = \
                self.forward_eval(embeddings, thresholds)

            for i in range(images.shape[0]):
                height, width = images[i].shape[1:]
                predicted_locs[i][:, [0, 2]] = predicted_locs[i][:, [0, 2]] * width
                predicted_locs[i][:, [1, 3]] = predicted_locs[i][:, [1, 3]] * height
                # mask = predicted_labels[i] != (len(self.classes) - 1)
                # predicted_locs[i] = predicted_locs[i][mask]
                # predicted_labels[i] = predicted_labels[i][mask]
                # predicted_scores[i] = predicted_scores[i][mask]
                # predicted_embeddings[i] = predicted_embeddings[i][mask]
            return predicted_locs, predicted_labels, predicted_scores, predicted_embeddings

    def find_pos_locs(self, boxes: List):
        batch_size = len(boxes)
        true_locs = list()

        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i].to(self.device),
                                           self.priors_xy.to(self.device))  # (n_objects, 8732)

            # For each prior, find the object that has the maximum overlap
            _, object_for_each_prior = overlap.max(dim=0)  # (8732)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)
            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(self.device)

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs.append(prior_for_each_object)
        return true_locs

    def create_prior_boxes(self):
        """
        Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.

        :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
        """
        fmap_dims = {'conv4_3': 38,
                     'conv7': 19,
                     'conv8_2': 10,
                     'conv9_2': 5,
                     'conv10_2': 3,
                     'conv11_2': 1}

        obj_scales = {'conv4_3': 0.1,
                      'conv7': 0.2,
                      'conv8_2': 0.375,
                      'conv9_2': 0.55,
                      'conv10_2': 0.725,
                      'conv11_2': 0.9}

        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                        # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                        # scale of the current feature map and the scale of the next feature map
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # For the last feature map, there is no "next" feature map
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(self.device)  # (8732, 4)
        prior_boxes.clamp_(0, 1)  # (8732, 4)

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, embeddings, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param embeddings: embeddings for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, emb_dim)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = len(predicted_locs)
        emb_dim = embeddings.shape[2]
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()
        all_image_embeddings = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1) == embeddings.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4), these are fractional pt. coordinates

            sample_embeddings = embeddings[i]

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()
            image_embeddings = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)
                class_embeddings = sample_embeddings[score_above_min_score]

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)
                class_embeddings = class_embeddings[sort_ind]  # (n_min_score, emb_dim)

                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(self.device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                rev_suppress = (1 - suppress).bool()
                image_boxes.append(class_decoded_locs[rev_suppress])
                image_labels.append(torch.LongTensor(rev_suppress.sum().item() * [c]).to(self.device))
                image_scores.append(class_scores[rev_suppress])
                image_embeddings.append(class_embeddings[rev_suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(self.device))
                image_labels.append(torch.LongTensor([0]).to(self.device))
                image_scores.append(torch.FloatTensor([0.]).to(self.device))
                image_embeddings.append(torch.FloatTensor([0.] * emb_dim).to(self.device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            image_embeddings = torch.cat(image_embeddings, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)
                image_embeddings = image_embeddings[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)
            all_image_embeddings.append(image_embeddings)

        return all_images_boxes, all_images_labels, all_images_scores, all_image_embeddings  # lists of length batch_size
