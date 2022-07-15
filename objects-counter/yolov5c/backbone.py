from layers import *
from torch import nn


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        #model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
        #torch.save(model.state_dict(), 'weights.pts')
        self.weights = torch.load('weights.pts')
        self.outs = []
        self.conv1 = Conv(3, 64, k=6, s=2, p=2)
        self.conv2 = Conv(64, 128, k=3, s=2)
        self.bottleneck1 = CSPBottleneck3CV(128, 128, bottlenecks_n=3, e=0.5)
        self.conv3 = Conv(128, 256, k=3, s=2)
        self.bottleneck2 = CSPBottleneck3CV(256, 256, bottlenecks_n=6, e=0.5, save_copy=self.outs)
        self.conv4 = Conv(256, 512, k=3, s=2)
        self.bottleneck3 = CSPBottleneck3CV(512, 512, bottlenecks_n=9, e=0.5, save_copy=self.outs)
        self.conv5 = Conv(512, 1024, k=3, s=2)
        self.bottleneck4 = CSPBottleneck3CV(1024, 1024, bottlenecks_n=3, e=0.5)
        self.SPP = SPPF(1024, 1024)
        self.model = nn.Sequential(
            self.conv1,  # 1
            self.conv2,  # 1
            self.bottleneck1,  # 9
            self.conv3,  # 1
            self.bottleneck2,  # 15
            self.conv4,  # 1
            self.bottleneck3,  # 21
            self.conv5,  # 1
            self.bottleneck4,  # 9
            self.SPP  # 2
        )
        # total: 61
        weight_index = 0
        weights_list = [_ for _ in self.weights]
        conv_index = 0
        for layer in self.model:
            #print('----------------<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>----------------')
            #print('LAYER -', layer)
            #print('----------------<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>----------------')
            if isinstance(layer, Conv):
                layer.layers[0].weight = nn.Parameter(self.weights[weights_list[weight_index]])
                layer.layers[0].bias = nn.Parameter(self.weights[weights_list[weight_index + 1]])
                #if conv_index == 0:
                #    print('ORIG WEIGHT', nn.Parameter(self.weights[weights_list[weight_index]])[0][0][0][0])
                #    print('ORIG BIAS', nn.Parameter(self.weights[weights_list[weight_index + 1]]))
                conv_index += 1
                weight_index += 2
            if isinstance(layer, SPPF):
                layer.cv1.layers[0].weight = nn.Parameter(self.weights[weights_list[weight_index]])
                layer.cv1.layers[0].bias = nn.Parameter(self.weights[weights_list[weight_index+1]])
                layer.cv2.layers[0].weight = nn.Parameter(self.weights[weights_list[weight_index+2]])
                layer.cv2.layers[0].bias = nn.Parameter(self.weights[weights_list[weight_index+3]])
                weight_index += 4
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

    def forward(self, x: torch.Tensor):
        return self.model(x)
