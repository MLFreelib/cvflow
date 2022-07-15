import torch.nn as nn
import torch
import pkg_resources as pkg


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False):
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)
    s = f'{name}{minimum} required by YOLOv5, but {name}{current} is currently installed'
    if hard:
        assert result, s
    return result


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=1, g=1, save_copy=None):
        super().__init__()
        self.save_copy = save_copy
        self.layers = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, bias=False, groups=g),
            #nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        out = self.layers(x)
        #print('CONV WEIGHT', self.layers[0].weight[0][0][0][0])
        #print('CONV', out[0][0][2][0], out.dtype)
        if self.save_copy is not None:
            self.save_copy.append(out)
        return out


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        h = int(c2 * e)  # hidden channels
        self.h = h
        self.c1 = c1
        self.cv1 = Conv(c1, h, 1, 1, p=0)
        self.cv2 = Conv(h, c2, 3, 1, g=g)
        self.layers = nn.Sequential(
            self.cv1,
            self.cv2
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        self.add = True
        if self.add:
            sx = self.cv1(x)
            #print('BN OUT 1', sx[0][0][2][0], 'cv2:', self.cv2.layers[0])
            sx = self.cv2(sx)
            #print('BN OUT 2', sx[0][0][2][0])
            out = x + sx
            #out = x + self.layers(x)
        else:
            out = self.layers(x)
        return out
        #return x + self.layers(x) if self.add else self.layers(x)


class BottleneckCSP(nn.Module):
    def __init__(self, ch, bn_n=1, e=1):
        super().__init__()
        h = int(ch * e)
        self.cv1 = Conv(ch, h, 1, 1)
        self.cv2 = Conv(h, h, 1, 1, bias=False)
        self.cv3 = Conv(ch, h, 1, 1, bias=False)
        self.m = nn.Sequential(*(Bottleneck(h) for _ in range(bn_n)))
        self.layers = nn.Sequential(
            nn.BatchNorm2d(2 * h),
            nn.SiLU(),
            Conv(2 * h, ch, 1, 1)
        )

    def forward(self, x):
        x1 = self.cv1(x)
        m = self.m(x1)
        y_cat = torch.cat((self.cv2(m), self.cv3(x)), dim=1)
        return self.layers(y_cat)


class CSPBottleneck3CV(nn.Module):
    def __init__(self, c1, c2, bottlenecks_n=1, e=1, save_copy=None, pre_save=True):
        super().__init__()
        h = int(c2 * e)
        self.cv1 = Conv(c1, h, 1, 1, p=0)
        self.cv2 = Conv(c1, h, 1, 1, p=0)
        self.cv3 = Conv(2 * h, c2, 1, p=0)
        self.save_copy = save_copy
        self.pre_save = pre_save
        self.bn = bottlenecks_n
        self.m = nn.Sequential(*(Bottleneck(h, h, e=1) for _ in range(bottlenecks_n)))

    def forward(self, x):
        #print('CSP Start')
        x1 = self.cv1(x)
        m = self.m(x1)
        x2 = self.cv2(x)
        x_cat = torch.cat((m, x2), dim=1)
        out = self.cv3(x_cat)
        if self.save_copy is not None:
            if self.pre_save:
                self.save_copy.append(x)
            else:
                self.save_copy.append(out)
        #print('CSP', out[0][0][2][0])
        return out


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):
        super().__init__()
        h = c1 // 2
        self.cv1 = Conv(c1, h, 1, 1, p=0)
        self.cv2 = Conv(h * 4, c2, 1, 1, p=0)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x1, y1, y2, self.m(y2)], 1))


class Concat(nn.Module):
    def __init__(self, skip_in, dimension=1):
        super().__init__()
        self.skip_in = skip_in
        self.d = dimension

    def forward(self, x):
        sx = self.skip_in.pop(-1)
        return torch.cat((sx, x), self.d)


class Detect(nn.Module):  # TODO: rewrite
    def __init__(self, nc=80, anchors=(), ch=(256, 512, 1024), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)
        self.stride = [1, 1, 1]

    def forward(self, x):
        #print(len(x), x[0].shape, x[1].shape, x[2].shape)
        z = []  # inference output
        #print(len(x), self.training, self.nl)
        #print('BEFORE DETECT', x)
        self.training = False
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            #print(f'DETECT {i}', x[i][0][0][2][0])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()

                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                else:
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        shape = 1, self.na, ny, nx, 2
        if check_version(torch.__version__, '1.10.0'):
            yv, xv = torch.meshgrid(torch.arange(ny, device=d), torch.arange(nx, device=d), indexing='ij')
        else:
            yv, xv = torch.meshgrid(torch.arange(ny, device=d), torch.arange(nx, device=d))
        grid = torch.stack((xv, yv), 2).expand(shape).float()
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape).float()
        return grid, anchor_grid
