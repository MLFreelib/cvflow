from layers import *


class Head(nn.Module):
    def __init__(self, nc=75, anchors=None):
        super().__init__()
        #model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
        #torch.save(model.state_dict(), 'weights.pts')
        self.weights = torch.load('weights.pts')
        self.nc = nc
        self.outs = []
        self.conv_outs = []
        self.bn_outs = []
        if anchors:
            self.anchors = anchors
        else:
            self.anchors = ((10, 13, 16, 30, 33, 23),
                            (30, 61, 62, 45, 59, 119),
                            (116, 90, 156, 198, 373, 326))
        self.conv1 = Conv(1024, 512, p=0, save_copy=self.conv_outs)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.bnc31 = CSPBottleneck3CV(1024, 512, e=0.5, bottlenecks_n=3)
        self.conv2 = Conv(512, 256, p=0, save_copy=self.conv_outs)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.bnc32 = CSPBottleneck3CV(512, 256, e=0.5, bottlenecks_n=3, save_copy=self.bn_outs, pre_save=False)
        self.conv3 = Conv(256, 256, k=3, s=2, p=1)
        self.bnc33 = CSPBottleneck3CV(512, 512, e=0.5, bottlenecks_n=3, save_copy=self.bn_outs, pre_save=False)
        self.conv4 = Conv(512, 512, k=3, s=2, p=1)
        self.bnc34 = CSPBottleneck3CV(1024, 1024, e=0.5, bottlenecks_n=3, save_copy=self.bn_outs, pre_save=False)
        self.detect = Detect(anchors=self.anchors)
        self.model = nn.Sequential(
            self.conv1,  # 1
            self.upsample1,
            Concat(self.outs),
            self.bnc31,  # 9
            self.conv2,  # 1
            self.upsample2,
            Concat(self.outs),
            self.bnc32,  # 9
            self.conv3,  # 1
            Concat(self.conv_outs),
            self.bnc33,  # 9
            self.conv4,  # 1
            Concat(self.conv_outs),
            self.bnc34  # 9
        )
        weight_index = 122
        weights_list = [_ for _ in self.weights]
        for layer in self.model:
            if isinstance(layer, Conv):
                layer.layers[0].weight = nn.Parameter(self.weights[weights_list[weight_index]])
                layer.layers[0].bias = nn.Parameter(self.weights[weights_list[weight_index + 1]])
                weight_index += 2
            if isinstance(layer, CSPBottleneck3CV):
                layer.cv1.layers[0].weight = nn.Parameter(self.weights[weights_list[weight_index]])
                layer.cv1.layers[0].bias = nn.Parameter(self.weights[weights_list[weight_index+1]])
                layer.cv2.layers[0].weight = nn.Parameter(self.weights[weights_list[weight_index+2]])
                layer.cv2.layers[0].bias = nn.Parameter(self.weights[weights_list[weight_index+3]])
                layer.cv3.layers[0].weight = nn.Parameter(self.weights[weights_list[weight_index+4]])
                layer.cv3.layers[0].bias = nn.Parameter(self.weights[weights_list[weight_index+5]])
                weight_index += 6
                for i in layer.m:
                    i.cv1.layers[0].weight = nn.Parameter(self.weights[weights_list[weight_index]])
                    i.cv1.layers[0].bias = nn.Parameter(self.weights[weights_list[weight_index+1]])
                    i.cv2.layers[0].weight = nn.Parameter(self.weights[weights_list[weight_index+2]])
                    i.cv2.layers[0].bias = nn.Parameter(self.weights[weights_list[weight_index+3]])
                    weight_index += 4

        weight_index += 1
        self.detect.m[0].weight = nn.Parameter(self.weights[weights_list[weight_index]])
        self.detect.m[0].bias = nn.Parameter(self.weights[weights_list[weight_index + 1]])
        self.detect.m[1].weight = nn.Parameter(self.weights[weights_list[weight_index + 2]])
        self.detect.m[1].bias = nn.Parameter(self.weights[weights_list[weight_index + 3]])
        self.detect.m[2].weight = nn.Parameter(self.weights[weights_list[weight_index + 4]])
        self.detect.m[2].bias = nn.Parameter(self.weights[weights_list[weight_index + 5]])
        weight_index += 6

    def forward(self, x):
        self.model(x)
        x = self.detect(self.bn_outs)
        self.outs.clear()
        self.conv_outs.clear()
        self.bn_outs.clear()
        return x
