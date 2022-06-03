import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import models
import math
import argparse
import os
import matplotlib.image as mpimg
import numpy as np


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class EpochSingle(nn.Module):
    def __init__(self):
        super(EpochSingle, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            # nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.InstanceNorm2d(32),
            #nn.Dropout(0.5)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            # nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.InstanceNorm2d(64),

            #nn.Dropout(0.25)

        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            # nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.InstanceNorm2d(128),

            #nn.Dropout(0.25)

        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            # nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.25)

        )
        self.layer5 = nn.Sequential(
            nn.Linear(51200, 256),
            # nn.Linear(25*8*128, 1024),

            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)

        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)

        out = self.layer5(out)

        return out    
class Epoch(nn.Module):
    def __init__(self):
        super(Epoch, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.InstanceNorm2d(32),
            #nn.Dropout(0.5)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.InstanceNorm2d(64),

            #nn.Dropout(0.25)

        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.InstanceNorm2d(128),

            #nn.Dropout(0.25)

        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.25)

        )
        self.layer5 = nn.Sequential(
            nn.Linear(51200, 256),
            # nn.Linear(25*8*128, 1024),

            nn.ReLU(),
            nn.Dropout(0.5),

        )
        self.fc = nn.Sequential(
            nn.Linear(533, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        features = []
        for i in range(2):
            out = self.layer1(x[0][:,i,:,:])
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = out.reshape(out.size(0), -1)
            out = self.layer5(out)
            features.append(out)
        features.append(x[1])
        out = torch.cat(features, dim=1)
        out = self.fc(out)
        return out

class Resnet101(nn.Module):
    def __init__(self, pretrained=False):
        super(Resnet101, self).__init__()
        self.model = models.resnet101(pretrained=pretrained)
        if pretrained:
            for parma in self.model.parameters():
                parma.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
                            nn.BatchNorm1d(num_ftrs, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            nn.Dropout(p=0.25),
                            nn.Linear(in_features=2048, out_features=256, bias=True),
                            nn.ReLU()
                            )
        self.fc = nn.Sequential(
            nn.Linear(533, 1),
            nn.ReLU()
        )

    def forward(self, x):
        features = []
        for i in range(2):
            out = self.model(x[0][:,i,:,:])

            # out = self.conv_new(out)
            # out = self.model.avgpool(out)
            # out = torch.flatten(out, 1)
            # out = self.model.classifier(out)
            features.append(out)
        # out2 = self.speedlayer(x[1])

        features.append(x[1])
        out = torch.cat(features, dim=-1)
        out = self.fc(out)
        # out = torch.cat([out, out2], dim=-1)
        # out = self.fc2(out)
        return out        

class Vgg16(nn.Module):
    def __init__(self, pretrained=False):
        super(Vgg16, self).__init__()
        self.model = models.vgg16(pretrained=pretrained)
        if pretrained:
            for parma in self.model.parameters():
                parma.requires_grad = False
        self.conv_new = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.model.classifier = nn.Sequential(
            nn.BatchNorm1d(512*7*7),
            nn.Linear(512*7*7, 256)
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(2048, 1024),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(1024, 256)
        )

        self.fc = nn.Sequential(
            nn.Linear(533, 1),
            nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(256, 1)
        )
        # self.fc2 = nn.Linear(2, 1)
    def forward(self, x):
        features = []
        for i in range(2):
            out = self.model.features(x[0][:,i,:,:])

            out = self.conv_new(out)
            out = self.model.avgpool(out)
            out = torch.flatten(out, 1)
            out = self.model.classifier(out)
            features.append(out)
        # out2 = self.speedlayer(x[1])

        features.append(x[1])
        out = torch.cat(features, dim=-1)
        out = self.fc(out)
        # out = torch.cat([out, out2], dim=-1)
        # out = self.fc2(out)
        return out

class Vgg16Single(nn.Module):
    def __init__(self, pretrained=False):
        super(Vgg16Single, self).__init__()
        self.model = models.vgg16(pretrained=pretrained)
        if pretrained:
            for parma in self.model.parameters():
                parma.requires_grad = False
        self.conv_new = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.model.classifier = nn.Sequential(
            nn.BatchNorm1d(512*7*7),
            nn.Linear(512*7*7, 256)
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(2048, 1024),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(1024, 256)
        )

        self.fc = nn.Sequential(
            # nn.Linear(533, 1)
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        # self.fc2 = nn.Linear(2, 1)
    def forward(self, x):


        out = self.model.features(x)

        out = self.conv_new(out)
        out = self.model.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.model.classifier(out)
        out = self.fc(out)
        return out

def build_vgg16(pretrained=True):
    model = models.vgg16(pretrained=pretrained)
    if pretrained:
        # for name, child in model.features.named_children():
        #     # print(name)
        #     if int(name) <= 24:
        #         for params in child.parameters():
        #             params.requires_grad = False
        for parma in model.parameters():
            parma.requires_grad = False    
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(25088),
        nn.Linear(25088, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    )
    return model

class Resnet101Single(nn.Module):
    def __init__(self, pretrained=False):
        super(Resnet101Single, self).__init__()
        self.model = models.resnet101(pretrained=pretrained)
        if pretrained:
            for parma in self.model.parameters():
                parma.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
                          nn.BatchNorm1d(num_ftrs, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                          nn.Dropout(p=0.25),
                          nn.Linear(in_features=2048, out_features=256, bias=True),
                          nn.ReLU(),
                          nn.Linear(256, 1)

                         )

    def forward(self, x):
        return self.model(x)


def build_resnet101(pretrained=False):
    model = models.resnet101(pretrained=pretrained)
    if pretrained:
        for parma in model.parameters():
            parma.requires_grad = False    
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
                          nn.BatchNorm1d(num_ftrs, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                          nn.Dropout(p=0.25),
                          nn.Linear(in_features=2048, out_features=256, bias=True),
                          nn.ReLU(),
                          nn.Linear(256, 1)
                         )
    
    return model 

if __name__ == "__main__":
    model = EpochSingle()
    summary(model, (3, 224, 224 ), device='cpu')
    # device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    # model = CNN3D(sequence_len=5)
    # # model = RNN()
    # model.to(device)
    # summary(model, (5,3, 224, 224))