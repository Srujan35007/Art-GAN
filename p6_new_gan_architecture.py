import time
b = time.time()
from torchsummary import summary
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from torchvision import datasets, transforms
import torch.optim as optim
a = time.time()
print(f'Imports complete in {a-b} seconds')


def get_up_layer(in_channels, out_channels,
                 kernel_size, stride, padding, activated=True):
    if activated is True:
        Convt = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                            kernel_size, stride, padding=padding),
            nn.BatchNorm2d(out_channels, momentum=0.3),
            nn.LeakyReLU(1)
        )
    else:
        Convt = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                            kernel_size, stride, padding=padding),
            nn.BatchNorm2d(out_channels, momentum=0.3)
        )
    nn.init.kaiming_uniform_(Convt[0].weight)
    return Convt


def get_down_layer(in_channels, out_channels,
                   kernel_size, stride, padding, activated=True):
    if activated is True:
        Conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                    kernel_size, stride, padding=padding),
            nn.BatchNorm2d(out_channels, momentum=0.3),
            nn.ReLU()
        )
    else:
        Conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                    kernel_size, stride, padding=padding),
            nn.BatchNorm2d(out_channels, momentum=0.3)
        )
    nn.init.kaiming_uniform_(Conv[0].weight)
    return Conv


class Generator(nn.Module):
    def __init__(self, latent_dims=256, max_channels=512):
        super(Generator, self).__init__()
        self.latent_dims = latent_dims
        n = max_channels
        self.lin1 = nn.Linear(self.latent_dims, 625)
        self.convt2 = get_up_layer(1, n, 3, 1, 1)
        self.convt3 = get_up_layer(n, n, 3, 1, 1)
        self.convt4 = get_up_layer(n, n//2, 4, 2, 1)
        self.convt5 = get_up_layer(n//2, n, 3, 1, 1)
        self.convt6 = get_up_layer(n, n//2, 3, 1, 1)
        self.convt7 = get_up_layer(n//2, n//2, 3, 1, 1)
        self.convt8 = get_up_layer(n//2, n//4, 4, 2, 1)
        self.convt9 = get_up_layer(n//4, n//2, 3, 1, 1)
        self.convt10 = get_up_layer(n//2, n//4, 3, 1, 1)
        self.convt11 = get_up_layer(n//4, n//4, 3, 1, 1)
        self.convt12 = get_up_layer(n//4, n//8, 4, 2, 1)
        self.convt13 = get_up_layer(n//8, n//4, 3, 1, 1)
        self.convt14 = get_up_layer(n//4, n//8, 3, 1, 1)
        self.convt15 = get_up_layer(n//8, n//8, 3, 1, 1)
        self.convt16 = get_up_layer(n//8, n//16, 4, 2, 1)
        self.convt17 = get_up_layer(n//16, n//8, 3, 1, 1)
        self.convt18 = get_up_layer(n//8, n//16, 3, 1, 1)
        self.convt19 = get_up_layer(n//16, n//16, 3, 1, 1)
        self.convt20 = get_up_layer(n//16, 3, 3, 1, 1)
        print(f'Generated created.')

    def __call__(self, x):
        x = x.view(-1, self.latent_dims)
        x = self.lin1(x)
        x = x.view(-1, 1, 25, 25)
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


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}')
gen = Generator(max_channels=128).to(device)

summary(gen, (1, 256))
