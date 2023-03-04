from __future__ import print_function

import torch.utils.data
import math

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                  padding=pad, stride=stride, bias=False),
        nn.BatchNorm3d(out_planes))


class depthregression(nn.Module):
    def __init__(self, maxdepth):
        super(depthregression, self).__init__()
        # self.disp = Variable(torch.Tensor(np.reshape(np.array(range(1, 1+maxdepth)), [1, maxdepth, 1, 1])).cuda(),
        #                      requires_grad=False)
        self.disp = torch.arange(
            1, 1 + maxdepth, device='cuda', requires_grad=False).float()[None, :, None, None]

    def forward(self, x):
        # disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * self.disp, 1)
        return out


class depthregression_std(nn.Module):
    def __init__(self, maxdepth):
        super(depthregression_std, self).__init__()
        # self.disp = Variable(torch.Tensor(np.reshape(np.array(range(1, 1+maxdepth)), [1, maxdepth, 1, 1])).cuda(), requires_grad=False)
        self.disp = torch.arange(
            1, 1 + maxdepth, device='cuda', requires_grad=False).float()[None, :, None, None]

    def forward(self, x, predict):
        disp = (self.disp - predict[:, None, :, :]) ** 2

        out = torch.sum(x * disp, 1)
        out = torch.log(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

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

    def forward(self, x):
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[
                                                         2], output_skip.size()[3]), mode='bilinear')

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[
                                                         2], output_skip.size()[3]), mode='bilinear')

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[
                                                         2], output_skip.size()[3]), mode='bilinear')

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[
                                                         2], output_skip.size()[3]), mode='bilinear')

        output_feature = torch.cat(
            (output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


class PSMNet(nn.Module):
    def __init__(self, maxdisp=192, maxdepth=80, down=2, scale=1):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.down = down
        self.maxdepth = maxdepth
        self.scale = scale

        self.feature_extraction = feature_extraction()

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
        xx = (calib / (self.down * 4.))[:, None] / torch.arange(1, 1 + self.maxdepth // self.down,device='cuda').float()[None, :]
        new_D = int(self.maxdepth // self.down)
        xx = xx.view(B, 1, new_D).repeat(1, C, 1)
        xx = xx.view(B, C, new_D, 1)
        yy = torch.arange(0, C, device='cuda').view(-1, 1).repeat(1, new_D).float()
        yy = yy.view(1, C, new_D, 1).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), -1).float()

        vgrid = Variable(grid)

        # scale grid to [-1,1]
        vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0] / max(D - 1, 1) - 1.0
        vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1] / max(C - 1, 1) - 1.0

        output = nn.functional.grid_sample(x, vgrid).contiguous()
        output = output.view(B, H, W, C, new_D).transpose(1, 3).transpose(2, 4)
        return output.contiguous()

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

    # def off_regress(self, off):
    #     "Regress offsets in [0, 1] range"
    #     off = F.tanh(off) * 1.5
    #     off = torch.clamp(off, min=-1, max=1)*0.5 + 0.5
    #     return off

    def off_regress(self, off):
        "Regress offsets in [0, 1] range"
        off = F.tanh(off)
        off = torch.clamp(off, min=-0.5, max=0.5) + 0.5
        return off

    def forward(self, left, right, calib, out_std=False, out_cost_volume=False):

        refimg_fea = self.feature_extraction(left)
        targetimg_fea = self.feature_extraction(right)

        # matching
        cost = Variable(
            torch.cuda.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1] * 2, self.maxdisp // 4,
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
        cost = self.warp(cost, calib)

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        off1 = self.semantic1(out1)
        off2 = self.semantic2(out2) + off1
        off3 = self.semantic3(out3) + off2

        if out_cost_volume:
            return cost3

        cost3 = F.upsample(cost3, [self.maxdepth // self.scale, left.size()[2], left.size()[3]], mode='trilinear')
        off3 = F.upsample(off3, [self.maxdepth // self.scale, left.size()[2], left.size()[3]], mode='trilinear')

        cost3 = torch.squeeze(cost3, 1)
        off3 = torch.squeeze(off3, 1)
        pred3 = F.softmax(cost3, dim=1)
        # off3 = F.sigmoid(off3)
        off3 = self.off_regress(off3)

        # For your information: This formulation 'softmax(c)' learned "similarity"
        # while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        # However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.

        _, pred3_out = torch.max(pred3, 1)
        pred3_out = pred3_out.float() + 1  # Make 1-indexed
        off3_out = self.interpolate_value(off3, pred3_out, maxdepth=self.maxdepth // self.scale)

        if out_std:
            return (pred3_out + off3_out) * self.scale, off3_out * self.scale
        return (pred3_out + off3_out) * self.scale