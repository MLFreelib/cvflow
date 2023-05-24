import math
from collections import OrderedDict
from enum import Enum
from functools import partial
from torch.cuda import amp

from torch import nn
import torch

from models.blocks import OutputFormat
from models.layers import Conv2dAuto, convbn, convbn_3d
from models.preprocessing import preprocess_for_YOLO
import torch.nn.functional as F
from torch.autograd import Variable

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dummy = nn.Parameter(torch.empty(0))
        self.in_channels, self.out_channels = in_channels, out_channels
        self._block = nn.Identity()

    @property
    def device(self):
        return self.dummy.device

    def forward(self, x, **kwargs):
        return self._block(x, **kwargs)


class MobileV1ResidualBlock(Block):
    expansion = 1

    def __init__(self, in_channels, planes, stride, downsample, pad, dilation, out_channels=None):
        super(MobileV1ResidualBlock, self).__init__(in_channels, out_channels)

        self.stride = stride
        self.downsample = downsample
        self._block = nn.ModuleList([self.downsample,
                                     self.convbn_dws(in_channels, planes, 3, stride, pad, dilation),
                                     self.convbn_dws(planes, planes, 3, 1, pad, dilation, second_relu=False)])

    def convbn_dws(self, inp, oup, kernel_size, stride, pad, dilation, second_relu=True):
        if second_relu:
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad,
                          dilation=dilation, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=False)
            )
        else:
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad,
                          dilation=dilation, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            )

    def forward(self, x):
        out = self._block[1](x)
        out = self._block[2](out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

class OutputBlock(Block):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)



class MobileV2ResidualBlock(Block):
    def __init__(self, in_channels, out_channels, stride, expanse_ratio, dilation=1):
        super(MobileV2ResidualBlock, self).__init__(in_channels, out_channels)
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(in_channels * expanse_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        pad = dilation

        if expanse_ratio == 1:
            self._block = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self._block = nn.Sequential(
                # pw
                nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self._block(x)
        else:
            return self._block(x)


class MobileStereoFeatureExtractionBlock(Block):
    def __init__(self, add_relus=False, in_channels=None, out_channels=None):
        super(MobileStereoFeatureExtractionBlock, self).__init__(in_channels, out_channels)

        self.expanse_ratio = 3
        self.inplanes = 32
        if add_relus:
            self.firstconv = nn.Sequential(MobileV2ResidualBlock(3, 32, 2, self.expanse_ratio),
                                           nn.ReLU(inplace=True),
                                           MobileV2ResidualBlock(32, 32, 1, self.expanse_ratio),
                                           nn.ReLU(inplace=True),
                                           MobileV2ResidualBlock(32, 32, 1, self.expanse_ratio),
                                           nn.ReLU(inplace=True)
                                           )
        else:
            self.firstconv = nn.Sequential(MobileV2ResidualBlock(3, 32, 2, self.expanse_ratio),
                                           MobileV2ResidualBlock(32, 32, 1, self.expanse_ratio),
                                           MobileV2ResidualBlock(32, 32, 1, self.expanse_ratio)
                                           )

        self.layer1 = self._make_layer(MobileV1ResidualBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(MobileV1ResidualBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(MobileV1ResidualBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(MobileV1ResidualBlock, 128, 3, 1, 1, 2)
        self._block = nn.Sequential(
            self.firstconv,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        )

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = [block(self.inplanes, planes, stride, downsample, pad, dilation)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self._block[0](x)
        x = self._block[1](x)
        l2 = self._block[2](x)
        l3 = self._block[3](l2)
        l4 = self._block[4](l3)
        feature_volume = torch.cat((l2, l3, l4), dim=1)

        return feature_volume


def load_weights(block, weight_index, weights_list, weights, bias=None, running=None):
        block.weight = nn.Parameter(weights[weights_list[weight_index]])
        weight_index += 1
        if bias is not None:
            block.bias = nn.Parameter(weights[weights_list[weight_index]])
            weight_index += 1
        if running is not None:
            block.running_mean = nn.Parameter(weights[weights_list[weight_index]], requires_grad=False)
            block.running_var = nn.Parameter(weights[weights_list[weight_index + 1]], requires_grad=False)
            block.num_batches_tracked = nn.Parameter(weights[weights_list[weight_index + 2]], requires_grad=False)
            weight_index += 3
        return block, weight_index

def import_weights(block, weight_index, weights_list, weights):
    try:
        for layer in block:
            if isinstance(layer, (nn.Sequential, nn. ModuleList)):
                layer, weight_index = import_weights(layer, weight_index, weights_list, weights)
            elif isinstance(layer, (MobileStereoNetInputBlock, MobileStereoFeatureExtractionBlock,
                                    MobileV2ResidualBlock, MobileV1ResidualBlock, MobileStereoNetBackbone,
                                    hourglass2D, GANetBasicBlock, hourglass)):
                 layer._block, weight_index = import_weights(layer._block, weight_index, weights_list, weights)
            else:
                try:
                    _ = layer.weight
                except AttributeError:
                    continue
                try:
                    running = layer.running_mean
                except AttributeError:
                    running = None
                bias = layer.bias
                layer, weight_index = load_weights(layer, weight_index, weights_list, weights, bias, running)
    except RuntimeError as e:
        print(e, weights_list[weight_index])

    return block, weight_index
class MobileStereoNetInputBlock(Block):
    def __init__(self, in_channels=3, out_channels=32):
        super().__init__(in_channels, out_channels)
        self.weights = None
        self.weight_index = 0
        self._block = nn.Sequential(
            MobileStereoFeatureExtractionBlock(add_relus=True),
            nn.Sequential(self.convbn(320, 256, 1, 1, 0, 1),
                          nn.ReLU(inplace=True),
                          self.convbn(256, 128, 1, 1, 0, 1),
                          nn.ReLU(inplace=True),
                          self.convbn(128, 64, 1, 1, 0, 1),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(64, 32, 1, 1, 0, 1))
        )

    def import_weights(self, weights, weight_index=0):
        self.weight_index = weight_index
        self.weights = weights
        weights_list = [_ for _ in self.weights]
        self._block, self.weight_index = import_weights(self._block, self.weight_index,
                                                        weights_list, self.weights)
        return  self.weight_index


    def convbn(self, in_channels, out_channels, kernel_size, stride, pad, dilation):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        L, R = x
        return self._block(L), self._block(R), L


class hourglass2D(Block):
    def __init__(self, in_channels, out_channels=None):
        super(hourglass2D, self).__init__(in_channels, out_channels)

        self.expanse_ratio = 2

        self.conv1 = MobileV2ResidualBlock(in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)

        self.conv2 = MobileV2ResidualBlock(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

        self.conv3 = MobileV2ResidualBlock(in_channels * 2, in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)

        self.conv4 = MobileV2ResidualBlock(in_channels * 4, in_channels * 4, stride=1, expanse_ratio=self.expanse_ratio)

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels))

        self.redir1 = MobileV2ResidualBlock(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
        self.redir2 = MobileV2ResidualBlock(in_channels * 2, in_channels * 2, stride=1,
                                            expanse_ratio=self.expanse_ratio)

        self._block = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.conv6,
            self.redir1,
            self.redir2
        )

    def forward(self, x):
        conv1 = self._block[0](x)
        conv2 =  self._block[1](conv1)

        conv3 =  self._block[2](conv2)
        conv4 = self._block[3](conv3)

        conv5 = F.relu(self._block[4](conv4) + self._block[7](conv2), inplace=True)
        conv6 = F.relu(self._block[5](conv5) + self._block[6](x), inplace=True)

        return conv6


class MobileStereoNetBackbone(Block):
    """
    Mobilestereonet backbone
    """

    def __init__(self, maxdisp=192, in_channels=1, out_channels=1, training=False, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.weights = None
        self.maxdisp = maxdisp

        self.num_groups = 1

        self.volume_size = 48

        self.hg_size = 48

        self.dres_expanse_ratio = 3

        self.weight_index = 686

        self.training = training

        self.conv3d = nn.Sequential(nn.Conv3d(1, 16, kernel_size=(8, 3, 3), stride=[8, 1, 1], padding=[0, 1, 1]),
                                    nn.BatchNorm3d(16),
                                    nn.ReLU(),
                                    nn.Conv3d(16, 32, kernel_size=(4, 3, 3), stride=[4, 1, 1], padding=[0, 1, 1]),
                                    nn.BatchNorm3d(32),
                                    nn.ReLU(),
                                    nn.Conv3d(32, 16, kernel_size=(2, 3, 3), stride=[2, 1, 1], padding=[0, 1, 1]),
                                    nn.BatchNorm3d(16),
                                    nn.ReLU())

        self.volume11 = nn.Sequential(self.convbn(16, 1, 1, 1, 0, 1),
                                      nn.ReLU(inplace=True))

        self.dres0 = nn.Sequential(MobileV2ResidualBlock(self.volume_size, self.hg_size, 1, self.dres_expanse_ratio),
                                   nn.ReLU(inplace=True),
                                   MobileV2ResidualBlock(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(MobileV2ResidualBlock(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio),
                                   nn.ReLU(inplace=True),
                                   MobileV2ResidualBlock(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio))

        self.encoder_decoder1 = hourglass2D(self.hg_size)

        self.encoder_decoder2 = hourglass2D(self.hg_size)

        self.encoder_decoder3 = hourglass2D(self.hg_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        self._block = nn.Sequential(
            self.conv3d,
            self.volume11,
            self.dres0,
            self.dres1,
            self.encoder_decoder1,
            self.encoder_decoder2,
            self.encoder_decoder3
        )

    def import_weights(self, weights, weight_index=0):
        self.weight_index = weight_index
        self.weights = weights
        weights_list = [_ for _ in self.weights]
        self._block, self.weight_index = import_weights(self._block, self.weight_index,
                                                        weights_list, self.weights)
        return  self.weight_index

    def convbn(self, in_channels, out_channels, kernel_size, stride, pad, dilation):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def interweave_tensors(self, refimg_fea, targetimg_fea):
        B, C, H, W = refimg_fea.shape
        interwoven_features = refimg_fea.new_zeros([B, 2 * C, H, W])
        interwoven_features[:, ::2, :, :] = refimg_fea
        interwoven_features[:, 1::2, :, :] = targetimg_fea
        interwoven_features = interwoven_features.contiguous()
        return interwoven_features

    def forward(self, x):

        featL, featR, L = x
        B, C, H, W = featL.shape
        volume = featL.new_zeros([B, self.num_groups, self.volume_size, H, W])
        for i in range(self.volume_size):
            if i > 0:
                x = self.interweave_tensors(featL[:, :, :, i:], featR[:, :, :, :-i])
                x = torch.unsqueeze(x, 1)
                x = self._block[0](x)
                x = torch.squeeze(x, 2)
                x = self._block[1](x)
                volume[:, :, i, :, i:] = x
            else:
                x = self.interweave_tensors(featL, featR)
                x = torch.unsqueeze(x, 1)
                x = self._block[0](x)
                x = torch.squeeze(x, 2)
                x = self._block[1](x)
                volume[:, :, i, :, :] = x

        volume = volume.contiguous()
        volume = torch.squeeze(volume, 1)
        cost0 = self._block[2](volume)
        cost0 = self._block[3](cost0) + cost0

        out1 = self._block[4](cost0)  # [2, hg_size, 64, 128]
        out2 = self._block[5](out1)
        out3 = self._block[6](out2)

        if self.training:
            return [cost0, out1, out2, out3], L
        else:
            return out3, L


class DepthOutput(OutputBlock):
    def __init__(self, maxdisp=192, training = False):
        super().__init__(in_channels=1, out_channels=1)
        self.weight_index = 1145
        self.weights = None
        self.hg_size = 48
        self.maxdisp = maxdisp
        self.volume_size = 48
        self.training = training

        self.classif0 = nn.Sequential(self.convbn(self.hg_size, self.hg_size, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1,
                                                bias=False, dilation=1))
        self.classif1 = nn.Sequential(self.convbn(self.hg_size, self.hg_size, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1,
                                                bias=False, dilation=1))
        self.classif2 = nn.Sequential(self.convbn(self.hg_size, self.hg_size, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1,
                                                bias=False, dilation=1))
        self.classif3 = nn.Sequential(self.convbn(self.hg_size, self.hg_size, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1,
                                                bias=False, dilation=1))
        self._block = nn.Sequential(self.classif3)

    def convbn(self, in_channels, out_channels, kernel_size, stride, pad, dilation):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def import_weights(self, weights, weight_index=0):
        self.weight_index = weight_index
        self.weights = weights
        weights_list = [_ for _ in self.weights]
        self._block, self.weight_index = import_weights(self._block, self.weight_index,
                                                        weights_list, self.weights)
        return self.weight_index


    def disparity_regression(self, x, maxdisp):
        assert len(x.shape) == 4
        disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
        disp_values = disp_values.view(1, maxdisp, 1, 1)
        return torch.sum(x * disp_values, 1, keepdim=False)

    def forward(self, x):

        if self.training:
            cost0, out1, out2, out3, L = x
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)
            cost3 = self.classif3(out3)

            cost0 = torch.unsqueeze(cost0, 1)
            cost0 = F.interpolate(cost0, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = self.disparity_regression(pred0, self.maxdisp)

            cost1 = torch.unsqueeze(cost1, 1)
            cost1 = F.interpolate(cost1, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = self.disparity_regression(pred1, self.maxdisp)

            cost2 = torch.unsqueeze(cost2, 1)
            cost2 = F.interpolate(cost2, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = self.disparity_regression(pred2, self.maxdisp)

            cost3 = torch.unsqueeze(cost3, 1)
            cost3 = F.interpolate(cost3, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = self.disparity_regression(pred3, self.maxdisp)

            return {OutputFormat.DEPTH.value: [pred0, pred1, pred2, pred3]}
        else:
            x, L = x

            cost3 = self._block(x)
            cost3 = torch.unsqueeze(cost3, 1)
            cost3 = F.interpolate(cost3, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = self.disparity_regression(pred3, self.maxdisp)

            return {OutputFormat.DEPTH.value: pred3}



class GANetBasicBlock(Block):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, downsample, pad, dilation):
        super(GANetBasicBlock, self).__init__(in_channels, out_channels)

        self.conv1 = nn.Sequential(convbn(in_channels, out_channels, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(out_channels, out_channels, 3, 1, pad, dilation)



        self.downsample = downsample
        self._block = nn.Sequential(self.conv1, self.conv2, self.downsample)
        self.stride = stride



    def forward(self, x):
        out = self._block[0](x)
        out = self._block[1](out)

        if self._block[2] is not None:
            x = self._block[2](x)

        out += x

        return out

class GANetInputBlock(Block):
    def __init__(self, calib = 1017., in_channels=None, out_channels=None):
        super(GANetInputBlock, self).__init__(in_channels, out_channels)
        self.inplanes = 32
        self.calib = calib
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(GANetBasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(GANetBasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(GANetBasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(GANetBasicBlock, 128, 3, 1, 1, 2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))

        self._block = nn.Sequential(self.firstconv, self.layer1,
                                    self.layer2, self.layer3, self.layer4,
                                    self.branch1,self.branch2, self.branch3,
                                    self.branch4, self.lastconv)

        self.weight_index = 0
        self.weights = None

    def import_weights(self, weights, weight_index=0):
        self.weight_index = weight_index
        self.weights = weights
        weights_list = [_ for _ in self.weights]
        self._block, self.weight_index = import_weights(self._block, self.weight_index,
                                                        weights_list, self.weights)
        return self.weight_index

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes,
                            stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def process(self, x):
        output = self._block[0](x)
        output = self._block[1](output)
        output_raw = self._block[2](output)
        output = self._block[3](output_raw)
        output_skip = self._block[4](output)

        output_branch1 = self._block[5](output_skip)
        output_branch1 = F.interpolate(output_branch1, (output_skip.size()[
                                                         2], output_skip.size()[3]), mode='bilinear')

        output_branch2 = self._block[6](output_skip)
        output_branch2 = F.interpolate(output_branch2, (output_skip.size()[
                                                         2], output_skip.size()[3]), mode='bilinear')

        output_branch3 = self._block[7](output_skip)
        output_branch3 = F.interpolate(output_branch3, (output_skip.size()[
                                                         2], output_skip.size()[3]), mode='bilinear')

        output_branch4 = self._block[8](output_skip)
        output_branch4 = F.interpolate(output_branch4, (output_skip.size()[
                                                         2], output_skip.size()[3]), mode='bilinear')

        output_feature = torch.cat(
            (output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self._block[9](output_feature)

        return output_feature

    def forward(self, x):
        L, R = x

        return self.process(L), self.process(R), L


class hourglass(Block):
    def __init__(self, in_channels, out_channels=None):
        super(hourglass, self).__init__(in_channels, out_channels)

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(in_channels * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(in_channels))  # +x

        self._block = nn.Sequential(self.conv1, self.conv2,
                                    self.conv3, self.conv4,
                                    self.conv5, self.conv6)

    def forward(self, x, presqu, postsqu):

        out = self._block[0](x)  # in:1/4 out:1/8
        pre = self._block[1](out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self._block[2](pre)  # in:1/8 out:1/16
        out = self._block[3](out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self._block[4](out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self._block[4](out) + pre, inplace=True)

        out = self._block[5](post)  # in:1/8 out:1/4

        return out, pre, post




class GANetBackbone(Block):

    def __init__(self, maxdisp=192, maxdepth=80, down=2, scale=1, calib=1017., in_channels = None, out_channels = None):
        super(GANetBackbone, self).__init__(in_channels, out_channels)
        self.maxdisp = maxdisp
        self.down = down
        self.maxdepth = maxdepth
        self.scale = scale
        self.weight_index = 0
        self.weights = None
        self.calib = calib

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self._block = nn.Sequential(self.dres0, self.dres1, self.dres2, self.dres3, self.dres4)

    def import_weights(self, weights, weight_index=0):
        self.weight_index = weight_index
        self.weights = weights
        weights_list = [_ for _ in self.weights]
        self._block, self.weight_index = import_weights(self._block, self.weight_index,
                                                        weights_list, self.weights)
        return self.weight_index

    def warp(self, x, calib):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, D, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        # B,C,D,H,W to B,H,W,C,D
        x = x.transpose(1, 3).transpose(2, 4)
        B, H, W, C, D = x.size()
        x = x.view(B, -1, C, D)
        # mesh grid
        calib = torch.FloatTensor([calib])
        xx = (calib / (self.down * 4.))[:, None] / torch.arange(1, 1 + self.maxdepth // self.down).float()[None, :]
        new_D = int(self.maxdepth // self.down)
        xx = xx.view(B, 1, new_D).repeat(1, C, 1)
        xx = xx.view(B, C, new_D, 1)
        yy = torch.arange(0, C).view(-1, 1).repeat(1, new_D).float()
        yy = yy.view(1, C, new_D, 1).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), -1).float()

        vgrid = Variable(grid)

        # scale grid to [-1,1]
        vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0] / max(D - 1, 1) - 1.0
        vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1] / max(C - 1, 1) - 1.0

        output = nn.functional.grid_sample(x, vgrid).contiguous()
        output = output.view(B, H, W, C, new_D).transpose(1, 3).transpose(2, 4)
        return output.contiguous()


    def forward(self, x, out_std=False, out_cost_volume=False):


        refimg_fea, targetimg_fea, left = x

        # matching
        cost = Variable(
            torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1] * 2, self.maxdisp // 4,
                                   refimg_fea.size()[2],
                                   refimg_fea.size()[3]).zero_())

        for i in range(self.maxdisp // 4):
            if i > 0:
                cost[:, :refimg_fea.size()[1], i, :, i:] = refimg_fea[:, :, :, i:]
                cost[:, refimg_fea.size()[1]:, i, :, i:] = targetimg_fea[:, :, :, :-i]
            else:
                cost[:, :refimg_fea.size()[1], i, :, :] = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :, :] = targetimg_fea
        cost = cost.contiguous()
        cost = self.warp(cost, float(self.calib))

        cost0 = self._block[0](cost)
        cost0 = self._block[1](cost0) + cost0

        out1, pre1, post1 = self._block[2](cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self._block[3](out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self._block[4](out2, pre1, post2)
        out3 = out3 + cost0

        return out1, out2, out3, left



class GANetOutputBlock(OutputBlock):
    def __init__(self, maxdisp=192, maxdepth=80, down=2, scale=1, in_channels=None, out_channels=None):
        super(GANetOutputBlock, self).__init__(in_channels, out_channels)
        self.maxdisp = maxdisp
        self.down = down
        self.maxdepth = maxdepth
        self.scale = scale
        self.weight_index = 0
        self.weights = None

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.semantic1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.semantic2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.semantic3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self._block = nn.Sequential(self.classif1, self.classif2,self.classif3,
                                    self.semantic1, self.semantic2, self.semantic3)

    def import_weights(self, weights, weight_index=0):
        self.weight_index = weight_index
        self.weights = weights
        weights_list = [_ for _ in self.weights]
        self._block, self.weight_index = import_weights(self._block, self.weight_index,
                                                        weights_list, self.weights)
        return  self.weight_index
    def off_regress(self, off):
        "Regress offsets in [0, 1] range"
        off = torch.tanh(off)
        off = torch.clamp(off, min=-0.5, max=0.5) + 0.5
        return off

    def interpolate_value(self, x, indices, maxdepth):
        """
        bilinear interpolate tensor x at sampled indices
        x: [B, D, H, W] (features)
        val: [B, H, W] sampled indices (1-indexed)
        """

        # B,D,H,W to B,H,W,D
        x = x.permute(0, 2, 3, 1)
        indices = torch.unsqueeze(indices - 1, -1)

        indices = torch.clamp(indices, 0, maxdepth - 1)
        idx0 = torch.floor(indices).long()
        idx1 = torch.min(idx0 + 1, (maxdepth - 1) * torch.ones_like(idx0))
        idx0 = torch.max(idx1 - 1, torch.zeros_like(idx0))

        y0 = torch.gather(x, -1, idx0)
        y1 = torch.gather(x, -1, idx1)

        lmbda = indices - idx0.float()
        output = (1 - lmbda) * y0 + (lmbda) * y1

        output = torch.squeeze(output, -1)
        return output

    def forward(self, x,  out_std=False, out_cost_volume=False):

        out1, out2, out3, left = x

        cost1 = self._block[0](out1)
        cost2 = self._block[1](out2) + cost1
        cost3 = self._block[2](out3) + cost2

        off1 = self._block[3](out1)
        off2 = self._block[4](out2) + off1
        off3 = self._block[5](out3) + off2

        if out_cost_volume:
            return cost3

        cost3 = F.interpolate(cost3, [self.maxdepth // self.scale, left.size()[2], left.size()[3]], mode='trilinear')
        off3 = F.interpolate(off3, [self.maxdepth // self.scale, left.size()[2], left.size()[3]], mode='trilinear')

        cost3 = torch.squeeze(cost3, 1)
        off3 = torch.squeeze(off3, 1)
        pred3 = F.softmax(cost3, dim=1)
        off3 = self.off_regress(off3)

        _, pred3_out = torch.max(pred3, 1)
        pred3_out = pred3_out.float() + 1  # Make 1-indexed
        off3_out = self.interpolate_value(off3, pred3_out, maxdepth=self.maxdepth // self.scale)

        if out_std:
            pred = (pred3_out + off3_out) * self.scale, off3_out * self.scale


        pred= (pred3_out + off3_out) * self.scale

        return {OutputFormat.DEPTH.value: pred}