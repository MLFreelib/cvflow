import torch
import torch.nn as nn


class BboxExpandNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.__res_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
        self.net = nn.ModuleDict({
            "res_net": self.__res_net,
            "res_net_out_ReLU": nn.ReLU(),
            "fc1": nn.Linear(self.__res_net.fc.out_features, 500),
            "fc1_out_ReLU": nn.ReLU(),
            "fc2": nn.Linear(500, 100),
            "fc2_out_ReLU": nn.ReLU(),
            "fc_resize": nn.Linear(100, 2),
            "sig_resize": nn.Sigmoid(),
            "fc_translate": nn.Linear(100, 2),
            "tanh_translate": nn.Tanh()
        })

    def forward(self, x):
        res_net_out = self.net.res_net_out_ReLU(self.net.res_net(x))

        fc1_out = self.net.fc1_out_ReLU(self.net.fc1(res_net_out))
        fc2_out = self.net.fc2_out_ReLU(self.net.fc2(fc1_out))

        resize_out = self.net.fc_resize(fc2_out)
        resize_out = self.net.sig_resize(resize_out)

        translate_out = self.net.fc_translate(fc2_out)
        translate_out = self.net.tanh_translate(translate_out)

        x = torch.cat((resize_out, translate_out), 1)

        return x

    def set_grad_enabled(self, mode):
        for _, module in self.net.items():
            self.__set_submodule_enabled(module, False)

        if mode:
            for module_name, module in self.net.items():
                if module_name != "res_net":
                    self.__set_submodule_enabled(module, True)

            self.__set_submodule_enabled(self.net.res_net.layer4[2], True)
            self.__set_submodule_enabled(self.net.res_net.avgpool, True)
            self.__set_submodule_enabled(self.net.res_net.fc, True)

    @staticmethod
    def __set_submodule_enabled(module, mode):
        for param in module.parameters():
            param.requires_grad = mode
