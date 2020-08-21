import time
b = time.time()
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchsummary import summary
a = time.time()
print(f'Imports complete in {a-b} seconds')


def get_up_layer(in_channels, out_channels, kernel_size, stride, padding):
    Convt = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels,
                           kernel_size, stride, padding=padding),
        nn.BatchNorm2d(out_channels, momentum=0.3),
        nn.LeakyReLU(0.02)
    )
    nn.init.kaiming_uniform_(Convt[0].weight)
    return Convt


def get_down_layer(in_channels, out_channels, kernel_size, stride, padding):
    Conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels,
                  kernel_size, stride, padding=padding),
        nn.BatchNorm2d(out_channels, momentum=0.3),
        nn.ReLU()
    )
    nn.init.kaiming_uniform_(Conv[0].weight)
    return Conv


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.lin1 = nn.Linear(256, 625)
        self.convt2 = get_up_layer(1, 512, 3, 1, 1)
        self.convt3 = get_up_layer(512, 512, 3, 1, 1)
        self.convt4 = get_up_layer(512, 256, 4, 2, 1)
        self.convt5 = get_up_layer(256, 512, 3, 1, 1)
        self.convt6 = get_up_layer(512, 256, 3, 1, 1)
        self.convt7 = get_up_layer(256, 256, 3, 1, 1)
        self.convt8 = get_up_layer(256, 128, 4, 2, 1)
        self.convt9 = get_up_layer(128, 256, 3, 1, 1)
        self.convt10 = get_up_layer(256, 128, 3, 1, 1)
        self.convt11 = get_up_layer(128, 128, 3, 1, 1)
        self.convt12 = get_up_layer(128, 64, 4, 2, 1)
        self.convt13 = get_up_layer(64, 128, 3, 1, 1)
        self.convt14 = get_up_layer(128, 64, 3, 1, 1)
        self.convt15 = get_up_layer(64, 64, 3, 1, 1)
        self.convt16 = get_up_layer(64, 32, 4, 2, 1)
        self.convt17 = get_up_layer(32, 64, 3, 1, 1)
        self.convt18 = get_up_layer(64, 32, 3, 1, 1)
        self.convt19 = get_up_layer(32, 32, 3, 1, 1)
        self.convt20 = get_up_layer(32, 3, 3, 1, 1)
        print(f'Generated created.')
    
    def __call__(self, x):
        x = x.view(-1, 256)
        x = self.lin1(x)
        x = x.view(-1, 1,25,25)
        x = self.convt2(x)
        x = self.convt3(x)
        x = self.convt4(x)
        x = self.convt5(x)
        x = self.convt6(x)
        x = self.convt7(x)
        x = self.convt8(x)
        x = self.convt9(x)
        x = self.convt10(x)
        x = self.convt11(x)
        x = self.convt12(x)
        x = self.convt13(x)
        x = self.convt14(x)
        x = self.convt15(x)
        x = self.convt16(x)
        x = self.convt17(x)
        x = self.convt18(x)
        x = self.convt19(x)
        x = self.convt20(x)
        return x

device = torch.device('cuda:0')
print(f'Running on {device}')
gen = Generator().to(device)

summary(gen, (1,256))
