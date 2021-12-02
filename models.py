import torch 
import torch.nn as nn 
from torchsummary import summary


# Generator
class GeneratorBlock(nn.Module):
    def __init__(self, ins, outs, ksize, stride, pad):
        super(GeneratorBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(ins, outs, ksize, stride, pad),
            nn.BatchNorm2d(outs),
            nn.LeakyReLU(0.2),
        )

    def forward(self, block_x):
        return self.block(block_x)

class Generator(nn.Module):
    def __init__(self, img_channels=3, in_filters=4):
        init_dims = 256
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            GeneratorBlock(in_filters, init_dims//1, 4, 2, 1),
            GeneratorBlock(init_dims//1, init_dims//2, 4, 2, 1),
            GeneratorBlock(init_dims//2, init_dims//4, 4, 2, 1),
            GeneratorBlock(init_dims//4, init_dims//8, 4, 2, 1),
            GeneratorBlock(init_dims//8, img_channels, 4, 2, 1),
        )

    def forward(self, x):
        return self.layers(x)


# Discriminator
class DiscriminatorBlock(nn.Module):
    def __init__(self, ins, outs, ksize, stride, pad):
        super(DiscriminatorBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ins, outs, ksize, stride, pad),
            nn.BatchNorm2d(outs),
            nn.ReLU(),
        )

    def forward(self, block_x):
        return self.block(block_x)

class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super(Discriminator, self).__init__()
        init_dims = 16
        self.layers = nn.Sequential(
            DiscriminatorBlock(img_channels, init_dims*1, 4, 2, 1),
            DiscriminatorBlock(init_dims*1, init_dims*2, 4, 2, 1),
            DiscriminatorBlock(init_dims*2, init_dims*4, 4, 2, 1),
            DiscriminatorBlock(init_dims*4, init_dims*8, 4, 2, 1),
            DiscriminatorBlock(init_dims*8, init_dims*16, 4, 2, 1),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(init_dims*16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)

disc = Discriminator()
summary(disc, (3,512,512))
