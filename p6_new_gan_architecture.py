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
                 kernel_size, stride, padding, activated=False):
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
    def __init__(self, latent_dims=256, max_channels=512, activation_neg_slope = 1):
        super(Generator, self).__init__()
        self.latent_dims = latent_dims
        self.activation_slope = activation_neg_slope
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
        act = nn.LeakyReLU(self.activation_slope)

        x = x.view(-1, self.latent_dims)
        x = self.lin1(x)
        x = x.view(-1, 1, 25, 25)
        out_2_up1 = self.convt2(act(x))
        out_3 = self.convt3(act(out_2_up1))
        out_4_sum1 = self.convt4(act(out_3))
        out_5_up1 = self.convt5(act(out_4_sum1))
        out_6_up2 = self.convt6(act(out_5_up1+ self.upsample(out_2_up1))) # Up1
        out_7_sum1 = self.convt7(act(out_6_up2))
        out_8_sum2 = self.convt8(act(out_7_sum1+out_4_sum1)) # Sum1
        out_9_up2 = self.convt9(act(out_8_sum2))
        out_10_up3 = self.convt10(act(out_9_up2+ self.upsample(out_6_up2))) # Up2
        out_11_sum2 = self.convt11(act(out_10_up3))
        out_12_sum3 = self.convt12(act(out_11_sum2+ out_8_sum2)) # Sum2
        out_13_up3 = self.convt13(act(out_12_sum3))
        out_14_up4 = self.convt14(act(out_13_up3+ self.upsample(out_10_up3))) # Up3
        out_15_sum3 = self.convt15(act(out_14_up4))
        out_16_sum4 = self.convt16(act(out_15_sum3+out_12_sum3)) # Sum3
        out_17_up4 = self.convt17(act(out_16_sum4))
        out_18 = self.convt18(act(out_17_up4+ self.upsample(out_14_up4))) # Up4
        out_19_sum4 = self.convt19(act(out_18))
        out_20 = self.convt20(act(out_19_sum4+out_16_sum4)) # Sum4
        return nn.Tanh()(out_20)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}')


class Discriminator(nn.Module):
    def __init__(self, max_channels=256):
        super(Discriminator, self).__init__()
        n = max_channels
        self.input1 = get_down_layer(3, n//8, 3,1,1)       # 200,200
        self.conv2 = get_down_layer(n//8, n//8, 3,1,1)
        self.conv3 = get_down_layer(n//8, n//4, 4,2,1)     # 100,100
        self.conv4 = get_down_layer(n//4, n//4, 3,1,1)
        self.conv5 = get_down_layer(n//4, n//4, 3,1,1)
        self.conv6 = get_down_layer(n//4, n//2, 4,2,1)      # 50,50
        self.conv7 = get_down_layer(n//2, n//2, 3,1,1)
        self.conv8 = get_down_layer(n//2, n, 4,2,1)      # 25,25
        self.conv9 = get_down_layer(n, n, 3,1,1)      
        self.conv10 = get_down_layer(n, 1, 3,1,1)        # 25,25,1
        self.lin11 = nn.Linear(25*25, 100)
        self.out12 = nn.Linear(100, 1)
        print('Discriminator created.')
    
    def __call__(self, x):
        x = self.input1(x.view(-1,3,400,400))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.lin11(x.view(-1,625))
        return self.out12(x)


disc = Discriminator(max_channels=128)
gen = Generator(max_channels=128).to(device)
print('Generator summary')
summary(gen, (1, 256))
print('Discriminator summary')
summary(disc, (400,400,3))