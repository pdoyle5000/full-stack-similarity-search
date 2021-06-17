import torch.nn as nn
from torch import sigmoid
from torch.nn.functional import relu
from collections import namedtuple

LayerIO = namedtuple("LayerIO", ["in_chan", "out_chan"])
params = LayerIO(in_chan=3, out_chan=16)


class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(params.in_chan, params.out_chan, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(params.out_chan, params.out_chan * 2, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(params.out_chan * 2, params.out_chan * 4, 3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(params.out_chan * 4, params.out_chan * 8, 3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(2)

        # self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        # self.relu5 = nn.ReLU(inplace=True)
        # self.maxpool5 = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        # x = self.conv5(x)
        # x = self.relu5(x)
        # x = self.maxpool5(x)
        # print(f"Conv 5: {x.shape}")
        return x


class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        # self.deconv1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        # self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(
            params.out_chan * 8, params.out_chan * 4, 2, stride=2
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.deconv3 = nn.ConvTranspose2d(
            params.out_chan * 4, params.out_chan * 2, 2, stride=2
        )
        self.relu3 = nn.ReLU(inplace=True)

        self.deconv4 = nn.ConvTranspose2d(
            params.out_chan * 2, params.out_chan, 2, stride=2
        )
        self.relu4 = nn.ReLU(inplace=True)

        self.deconv5 = nn.ConvTranspose2d(params.out_chan, params.in_chan, 2, stride=2)
        self.relu5 = nn.ReLU(inplace=True)

    def forward(self, x):
        # x = self.deconv1(x)
        # x = self.relu1(x)

        x = self.deconv2(x)
        x = self.relu2(x)

        x = self.deconv3(x)
        x = self.relu3(x)

        x = self.deconv4(x)
        x = self.relu4(x)

        x = self.deconv5(x)
        x = self.relu5(x)
        return x
