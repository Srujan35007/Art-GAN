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

    def upsample(self, tensor_):
        return nn.Upsample(scale_factor=2, mode='nearest')(tensor_)

    def __call__(self, x):
        x = x.view(-1, self.latent_dims)
        x = self.lin1(x)
        x = x.view(-1, 1, 25, 25)
        out_2_up1 = self.convt2(x)
        out_3 = self.convt3(out_2_up1)
        out_4_sum1 = self.convt4(out_3)
        out_5_up1 = self.convt5(out_4_sum1)
        out_6_up2 = self.convt6(out_5_up1+ self.upsample(out_2_up1)) # Up1
        out_7_sum1 = self.convt7(out_6_up2)
        out_8_sum2 = self.convt8(out_7_sum1+out_4_sum1) # Sum1
        out_9_up2 = self.convt9(out_8_sum2)
        out_10_up3 = self.convt10(out_9_up2+ self.upsample(out_6_up2)) # Up2
        out_11_sum2 = self.convt11(out_10_up3)
        out_12_sum3 = self.convt12(out_11_sum2+ out_8_sum2) # Sum2
        out_13_up3 = self.convt13(out_12_sum3)
        out_14_up4 = self.convt14(out_13_up3+ self.upsample(out_10_up3)) # Up3
        out_15_sum3 = self.convt15(out_14_up4)
        out_16_sum4 = self.convt16(out_15_sum3+out_12_sum3) # Sum3
        out_17_up4 = self.convt17(out_16_sum4)
        out_18 = self.convt18(out_17_up4+ self.upsample(out_14_up4)) # Up4
        out_19_sum4 = self.convt19(out_18)
        out_20 = self.convt20(out_19_sum4+out_16_sum4) # Sum4
        return out_20


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}')
gen = Generator(max_channels=128).to(device)

summary(gen, (1, 256))
