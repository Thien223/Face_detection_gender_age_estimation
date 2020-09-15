# -*- coding:utf-8 -*-
import torch.nn as nn
from modules.classify import Classify, Classify_

cfg = {
    'vgg-s': [64,     'M', 128,      'M', 256, 256,           'M'],
    'vgg11': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(self, vgg_type, batch_norm=True):
        super().__init__()
        self.vgg_type = vgg_type
        self.batch_norm = batch_norm
        self.cnn = self.cnn_layers()
        input_size = 2048
        if vgg_type=='vgg-s':
            input_size = input_size * 8

        self.classify = Classify_(input_size=input_size)

    def forward(self, inputs):
        cnn_output = self.cnn(inputs)
        x = cnn_output.view(cnn_output.size(0), -1)
        x = self.classify(x)
        return x

    def cnn_layers(self):
        layers = []
        input_channel = 3 ### RGB image
        #### vgg 19 architecture: 2 conv_64 + max pooling + 2 conv_128 + max_pooling + 4 conv_256 + max_pooling + (4 conv_512 + max_pooling) * 2

        for output_channel in cfg[self.vgg_type]:
            if output_channel == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue

            layers += [
                nn.Conv2d(in_channels=input_channel, out_channels=output_channel,
                          kernel_size=(3, 3), padding=1)
            ]

            if self.batch_norm:
                layers += [nn.BatchNorm2d(output_channel)]
            layers += [nn.ReLU(inplace=True)]
            input_channel = output_channel

        return nn.Sequential(*layers)

if __name__ == '__main__':
    pass
