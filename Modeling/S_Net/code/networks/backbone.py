import torch
import torchvision
import torch.nn.modules
import torch.nn as nn
import torchvision.models as models

class vgg16_bn(nn.Module):
    def __init__(self, pretrained=False):
        super(vgg16_bn, self).__init__()

        model = list(torchvision.models.vgg16_bn(pretrained=pretrained).features.children())

        self.model1 = torch.nn.Sequential(*model[:33])
        self.model2 = torch.nn.Sequential(*model[34:44])

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(out1)
        return out1, out2

class vgg16(nn.Module):
    def __init__(self, pretrained=True):
        super(vgg16, self).__init__()

        model = models.vgg16(pretrained=pretrained)

        vgg_feature_layers = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2',
                              'pool1', 'conv2_1', 'relu2_1', 'conv2_2',
                              'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
                              'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3',
                              'pool3', 'conv4_1', 'relu4_1', 'conv4_2',
                              'relu4_2', 'conv4_3', 'relu4_3']

        last_layer = 'relu4_3'
        last_layer_idx = vgg_feature_layers.index(last_layer)

        self.model1 = nn.Sequential(*list(model.features.children())[:last_layer_idx + 1])
        self.model2 = nn.Sequential(*list(model.features.children())[last_layer_idx + 1:-1])

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(out1)

        return out1, out2

