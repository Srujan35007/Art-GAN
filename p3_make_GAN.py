import time 
b = time.time()
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
from torchsummary import summary
import numpy as np 
import matplotlib.pyplot as plt; plt.style.use('fivethirtyeight')
from tqdm.notebook import tqdm 
a = time.time()
print(f'Imports complete in {a-b} seconds')

# We use a ResNet with Conv2dTranspose for avoiding diminishing gradients
class Res_conv_transpose_block(nn.Module):
    def __init__(self, n_channels, skip = True, bias_ = True):
        super(Res_conv_transpose_block, self).__init__()
        self.skip = skip
        self.conv1 = nn.ConvTranspose2d(n_channels, n_channels, 3, 1, padding=1, bias=bias_)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv2 = nn.ConvTranspose2d(n_channels, n_channels, 3, 1, padding=1, bias=bias_)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.conv3 = nn.ConvTranspose2d(n_channels, n_channels, 3, 1, padding=1, bias=bias_)
        self.relu3 = nn.ReLU(inplace=True)
        self.init_weights()

    def __call__(self, x):
        if self.skip is True:
            residual = x
            x = self.bn1(self.relu1(self.conv1(x)))
            x = self.bn2(self.relu2(self.conv2(x)))
            return self.relu3(self.conv3(x + residual))
        else:
            x = self.bn1(F.relu(self.conv1(x)))
            x = self.bn2(F.relu(self.conv2(x)))
            return F.relu(self.conv3(x))

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(module.weight)

def conv_transpose_res_block(n_channels):
    return Res_conv_transpose_block(n_channels)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.lin_input = nn.Linear(100, 100)
        self.input = nn.ConvTranspose2d(1, 30, 4,2,padding=1)
        self.block_a1 = nn.Sequential(
            conv_transpose_res_block(30),
            nn.ReLU(inplace=True),
            conv_transpose_res_block(30),
            nn.ReLU(inplace=True)
        )
        self.convt_ab = nn.ConvTranspose2d(30, 20, 6,4,padding=1)
        self.block_b1 = nn.Sequential(
            conv_transpose_res_block(20),
            nn.ReLU(inplace=True),
            conv_transpose_res_block(20),
            nn.ReLU(inplace=True)
        )
        self.convt_bc = nn.ConvTranspose2d(20, 10, 7,5,padding=1)
        self.block_c1 = nn.Sequential(
            conv_transpose_res_block(10),
            nn.ReLU(inplace=True),
            conv_transpose_res_block(10),
            nn.ReLU(inplace=True)
        )
        self.output = nn.ConvTranspose2d(10, 3, 3,1,padding=1)
    
    def __call__(self, x):
        x = self.lin_input(x).view(-1, 1,10,10)
        x = F.relu(self.input(x))
        x = self.block_a1(x)
        x = F.relu((self.convt_ab(x)), inplace = True)
        x = self.block_b1(x)
        x = F.relu((self.convt_bc(x)), inplace = True)
        x = self.block_c1(x)
        x = F.relu(self.output(x), inplace=True)
        return x

net = Generator()
t = torch.tensor(np.random.normal(0.5, 1, size=100))
t = t.float()
print(net(t).shape) # out_shape = (3,400,400) channels, width, height
out = net(t).view(400,400,3).detach().numpy()
plt.imshow(out)
plt.show()
