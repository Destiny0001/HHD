import torch.nn as nn
from torchvision import models
import torch
from utils.PreResnet import *

LAYER1_NODE = 40960


def weights_init(m):
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias.data, 0.)

class HASH_Net(nn.Module):
    def __init__(self, model_name, bit, pretrained=True):
        super(HASH_Net, self).__init__()
        if model_name == "alexnet":
            original_model = models.alexnet(pretrained)
            self.features = original_model.features
            # self.features_i2t = original_model.features
            cl1 = nn.Linear(256 * 6 * 6, 4096)
            cl2 = nn.Linear(4096, 4096)
            cl3 = nn.Linear(4096, bit)
            if pretrained:
                cl1.weight = original_model.classifier[1].weight
                cl1.bias = original_model.classifier[1].bias
                cl2.weight = original_model.classifier[4].weight
                cl2.bias = original_model.classifier[4].bias
            self.classifier = nn.Sequential(
                nn.Dropout(),
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                cl3,
                nn.Tanh()
            )
            self.model_name = 'alexnet'

        elif model_name == "vgg11":
            original_model = models.vgg11(pretrained)
            self.features = original_model.features
            cl1 = nn.Linear(25088, 4096)

            cl2 = nn.Linear(4096, 4096)
            cl3 = nn.Linear(4096, bit)

            if pretrained:
                cl1.weight = original_model.classifier[0].weight
                cl1.bias = original_model.classifier[0].bias
                cl2.weight = original_model.classifier[3].weight
                cl2.bias = original_model.classifier[3].bias

            self.classifier = nn.Sequential(
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl3,
                nn.Tanh()
            )
            self.model_name = 'vgg11'

        elif model_name == "resnet34":
            #original_model = models.resnet34(pretrained)
            original_model = ResNet34()
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            #self.features= original_model.features
            # in_features depends on the output of the last convolution layer
            #in_features = original_model.fc.in_features
            in_features  = 8192
            size=1024
            cl1 = nn.Linear(in_features, size)
            cl2 = nn.Linear(size, size)
            cl3 = nn.Linear(size, bit)
            self.classifier = nn.Sequential(
                #nn.ReLU(inplace=True),
                nn.Dropout(),
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl3,
                nn.Tanh()
            )
            self.model_name = "resnet34"





    def forward(self, x):
        f = self.features(x)
        if self.model_name == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)
            f = self.classifier(f)
        elif self.model_name == 'resnet34':
            f = torch.flatten(f, 1) 
            #f = f.view(f.size(0), -1)
            f = self.classifier(f)
        else:
            f = f.view(f.size(0), -1)
            f = self.classifier(f)
        return f

