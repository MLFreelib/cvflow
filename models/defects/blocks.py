from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from models.blocks import Block, OutputFormat


class AuxiliaryConvolutions(Block):
    """
    Additional convolutions to produce higher-level feature maps.
    """

    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__(-1, -1)

        # Auxiliary/additional convolutions on top of the VGG base
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)  # stride = 1, by default
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv7_feats):
        """
        Forward propagation.

        :param conv7_feats: lower-level conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :return: higher-level feature maps conv8_2, conv9_2, conv10_2, and conv11_2
        """
        out = F.relu(self.conv8_1(conv7_feats))  # (N, 256, 19, 19)
        out = F.relu(self.conv8_2(out))  # (N, 512, 10, 10)
        conv8_2_feats = out  # (N, 512, 10, 10)

        out = F.relu(self.conv9_1(out))  # (N, 128, 10, 10)
        out = F.relu(self.conv9_2(out))  # (N, 256, 5, 5)
        conv9_2_feats = out  # (N, 256, 5, 5)

        out = F.relu(self.conv10_1(out))  # (N, 128, 5, 5)
        out = F.relu(self.conv10_2(out))  # (N, 256, 3, 3)
        conv10_2_feats = out  # (N, 256, 3, 3)

        out = F.relu(self.conv11_1(out))  # (N, 128, 3, 3)
        conv11_2_feats = F.relu(self.conv11_2(out))  # (N, 256, 1, 1)

        # Higher-level feature maps
        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats


def make_conv_layers(n_boxes: Dict, aux_add: int, aux_out: int):
    conv4_3 = nn.Conv2d(n_boxes['conv4_3'] * aux_add * 2, n_boxes['conv4_3'] * aux_out, kernel_size=3, padding=1)
    conv7 = nn.Conv2d(n_boxes['conv7'] * aux_add * 2, n_boxes['conv7'] * aux_out, kernel_size=3, padding=1)
    conv8_2 = nn.Conv2d(n_boxes['conv8_2'] * aux_add * 2, n_boxes['conv8_2'] * aux_out, kernel_size=3, padding=1)
    conv9_2 = nn.Conv2d(n_boxes['conv8_2'] * aux_add * 2, n_boxes['conv9_2'] * aux_out, kernel_size=3, padding=1)
    conv10_2 = nn.Conv2d(n_boxes['conv8_2'] * aux_add * 2, n_boxes['conv10_2'] * aux_out, kernel_size=3, padding=1)
    conv11_2 = nn.Conv2d(n_boxes['conv8_2'] * aux_add * 2, n_boxes['conv11_2'] * aux_out, kernel_size=3, padding=1)
    return conv4_3, conv7, conv8_2, conv9_2, conv10_2, conv11_2


def make_emb_layers(n_boxes: Dict, aux_add: int):
    emb_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * aux_add, kernel_size=3, padding=1)
    emb_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * aux_add, kernel_size=3, padding=1)
    emb_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * aux_add, kernel_size=3, padding=1)
    emb_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * aux_add, kernel_size=3, padding=1)
    emb_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * aux_add, kernel_size=3, padding=1)
    emb_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * aux_add, kernel_size=3, padding=1)
    return emb_conv4_3, emb_conv7, emb_conv8_2, emb_conv9_2, emb_conv10_2, emb_conv11_2


class PredictionConvolutions(Block):
    """
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.

    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 8732 prior (default) boxes.
    See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.

    The class scores represent the scores of each object class in each of the 8732 bounding boxes located.
    A high score for 'background' = no object.
    """

    def __init__(self, emb_dim: int = 512):
        """
        :param n_classes: number of different types of objects
        """
        super(PredictionConvolutions, self).__init__(-1, -1)

        self.emb_dim = emb_dim

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {'conv4_3': 4,
                   'conv7': 6,
                   'conv8_2': 6,
                   'conv9_2': 6,
                   'conv10_2': 4,
                   'conv11_2': 4}

        # Embedding prediction convolutions (predict feature embeddings in localization boxes)
        self.emb_conv4_3, self.emb_conv7, self.emb_conv8_2, self.emb_conv9_2, self.emb_conv10_2, self.emb_conv11_2 = \
            make_emb_layers(n_boxes=n_boxes, aux_add=emb_dim)

        # 4 prior-boxes implies we use 4 different aspect ratios, etc.
        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.loc_layer = nn.Linear(in_features=emb_dim * 2, out_features=4)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward_embeddings(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats,
                           conv11_2_feats):
        batch_size = conv4_3_feats.size(0)

        emb_conv4_3 = self.emb_conv4_3(conv4_3_feats)  # (N, 4 * emb_dim, 38, 38)
        emb_conv4_3 = emb_conv4_3.permute(0, 2, 3,
                                          1).contiguous()  # (N, 38, 38, 4 * emb_dim), to match prior-box order (after .view())
        emb_conv4_3 = emb_conv4_3.view(batch_size, -1,
                                       self.emb_dim)  # (N * 5776, emb_dim), there are a total 5776 boxes on this feature map

        emb_conv7 = self.emb_conv7(conv7_feats)  # (N, 6 * emb_dim, 19, 19)
        emb_conv7 = emb_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 6 * emb_dim)
        emb_conv7 = emb_conv7.view(batch_size, -1,
                                   self.emb_dim)  # (N * 2166, n_classes), there are a total 2116 boxes on this feature map

        emb_conv8_2 = self.emb_conv8_2(conv8_2_feats)  # (N, 6 * emb_dim, 10, 10)
        emb_conv8_2 = emb_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 6 * emb_dim)
        emb_conv8_2 = emb_conv8_2.view(batch_size, -1, self.emb_dim)  # (N * 600, emb_dim)

        emb_conv9_2 = self.emb_conv9_2(conv9_2_feats)  # (N, 6 * emb_dim, 5, 5)
        emb_conv9_2 = emb_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 6 * emb_dim)
        emb_conv9_2 = emb_conv9_2.view(batch_size, -1, self.emb_dim)  # (N * 150, emb_dim)

        emb_conv10_2 = self.emb_conv10_2(conv10_2_feats)  # (N, 4 * emb_dim, 3, 3)
        emb_conv10_2 = emb_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 4 * emb_dim)
        emb_conv10_2 = emb_conv10_2.view(batch_size, -1, self.emb_dim)  # (N * 36, emb_dim)

        emb_conv11_2 = self.emb_conv11_2(conv11_2_feats)  # (N, 4 * emb_dim)
        emb_conv11_2 = emb_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 4 * emb_dim)
        emb_conv11_2 = emb_conv11_2.view(batch_size, -1, self.emb_dim)  # (N * 4, emb_dim)

        embeddings = torch.cat([emb_conv4_3, emb_conv7, emb_conv8_2, emb_conv9_2, emb_conv10_2, emb_conv11_2], dim=1)
        embeddings = torch.sigmoid(embeddings)
        return embeddings

    def forward(self, embeddings: torch.Tensor):

        # Predict classes in localization boxes
        # A total of 8732 boxes

        loc_out = self.loc_layer(embeddings)
        return loc_out


class OutBlock(Block):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)

    def forward(self, x, **kwargs):
        out = list()
        for i in range(len(x[0])):
            if len(x) == 4:
                out.append(
                    {
                        OutputFormat.BBOX.value: x[0][i],
                        'labels': x[1][i],
                        OutputFormat.CONFIDENCE.value: x[2][i],
                        'embeddings': x[3][i]
                    }
                )
            else:
                out.append(
                    {
                        OutputFormat.BBOX.value: x[0][i],
                        'embeddings': x[1][i]
                    }
                )
        return out
