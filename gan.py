# This contains both discriminator and generator nets
# without any residual connections

import time 
b = time.time()
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
from torchsummary import summary
import numpy as np 
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm 
a = time.time()
print(f'Imports complete in {a-b} seconds')


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.input = nn.Linear(100, 300) # 50 lin
        self.lin1 = nn.Linear(300, 625)   # 625 lin
        self.convt1 = nn.ConvTranspose2d(1, 200, 4,2,padding=1) # 50,50,200
        self.convt2 = nn.ConvTranspose2d(200, 150, 4,2,padding=1) # 100,100,150
        self.bn1 = nn.BatchNorm2d(150)
        self.convt3 = nn.ConvTranspose2d(150,100, 4,2,padding=1) # 200,200,100
        self.convt4 = nn.ConvTranspose2d(100,50, 4,2,padding=1) # 400,400,50
        self.bn2 = nn.BatchNorm2d(50)
        self.output = nn.ConvTranspose2d(50,3, 3,1,padding=1) # 400,400,3

        self.init_weights()
        self.optimizer = optim.Adam(self.parameters(), lr= 0.003)
    
    def __call__(self, x):
        x = F.relu(self.input(x.view(-1,100)))
        x = F.relu(self.lin1(x))
        x = x.view(-1,1,25,25)
        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))
        x = self.bn1(x)
        x = F.relu(self.convt3(x))
        x = F.relu(self.convt4(x))
        x = self.bn2(x)
        x = F.relu(self.output(x))
        return x.view(-1,3,400,400)

    def get_loss(self, disc_gen_img_out):
        ones = torch.ones_like(disc_gen_img_out)
        loss_fn = nn.BCELoss()
        return loss_fn(ones, disc_gen_img_out)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(module.weight)
        print('Generator weigths initialized')
    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.input = nn.Conv2d(3, 100, 3,1,padding=1) # 400,400,100
        self.conv1 = nn.Conv2d(100,150, 4,2,padding=1) # 200,200,150
        self.conv2 = nn.Conv2d(150,200, 4,2,padding=1) # 100,100,200
        self.bn1 = nn.BatchNorm2d(200)
        self.conv3 = nn.Conv2d(200,250, 4,2,padding=1) # 50,50,250
        self.conv4 = nn.Conv2d(250,300, 4,2,padding=1) # 25,25,300
        self.bn2 = nn.BatchNorm2d(300)
        self.conv5 = nn.Conv2d(300, 2, 3,1,padding=1) # 25,25,2
        self.lin1 = nn.Linear(25*25*2, 100) # lin 100
        self.out = nn.Linear(100, 1) # lin 1

        self.init_weights()
        self.optimizer = optim.Adam(self.parameters(), lr = 0.003)

    def __call__(self, x):
        x = x.view(-1,3,400,400)
        x = F.relu(self.input(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.bn1(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.bn2(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.lin1(x.view(-1,25*25*2)))
        x = torch.sigmoid(self.out(x))
        return x

    def get_loss(self, disc_gen_img_out, disc_real_img_out):
        loss_fn = nn.BCELoss()
        real_loss = loss_fn(torch.ones_like(disc_real_img_out), disc_real_img_out)
        fake_loss = loss_fn(torch.zeros_like(disc_gen_img_out), disc_gen_img_out)
        return (real_loss+fake_loss)/2

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
        print('Discriminator weigths initialized')
