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
import cv2
import os
from pathlib import Path
import random
a = time.time()
print(f'Imports complete in {a-b} seconds')


def make_batches(file_paths, batch_size):
    batch = []
    file_paths.sort()
    for i in range(len(file_paths)//batch_size):
        batch.append(random.choice(file_paths[(i*batch_size):((i+1)*batch_size)]))
    return batch

def plot_and_save(val_image_noise):
    # TO_DO
    pass

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

    def preprocess(self, image_path):
        image_array = cv2.imread(image_path)
        np_array = ((np.asarray(image_array)-127.5)/127.5).reshape(3,400,400)
        return torch.tensor(np_array).float().to(device)

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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')
gen = Generator().to(device)
disc = Discriminator().to(device)

# Get real image paths
data_path = f''
file_paths = []
for roots,dirs,files in os.walk(data_path):
    for file_ in files:
        file_paths.append(f'{data_path}/{file_}')
print(f'Total no. of file_paths = {len(file_paths)}')

# Make normal random noise for validation
val_image_noise = []
for i in range(16):
    noise = torch.tensor(np.random.normal(0,0.4, 100)).float().to(device)
    val_image_noise.append(noise)

gen_save_filename = 'art_gen.pt'
disc_save_filename = 'art_disc.pt'

gen_loss_list = []
disc_loss_list = []
disc_acc_list = []
val_image_path = './val_images'
train_flag = True
epoch_count = 0
# The training loop
while train_flag:
    epoch_count += 1
    train_batch = make_batches(file_paths, 30)
    temp_gen_loss_list = []
    temp_disc_loss_list = []
    correct, total = 0, 0
    for file_path in train_batch:
        gen.zero_grad()
        disc.zero_grad()
        noise = torch.tensor(np.random.normal(0,0.4, 100)).float().to(device)
        gen_img_out = gen(noise)
        disc_gen_img_out = disc(gen_img_out)
        disc_real_img_out = disc(disc.preprocess(file_path))
        gen_loss = gen.get_loss(disc_gen_img_out)
        disc_loss = disc.get_loss(disc_gen_img_out, disc_real_img_out)
        gen_loss.backward()
        disc_loss.backward()
        gen.optimizer.step()
        disc.optimizer.step()

        # For metrics
        temp_gen_loss_list.append(gen_loss.item())
        temp_disc_loss_list.append(disc_loss.item())
        if disc_gen_img_out.item() < 0.5:
            correct += 1
            total += 1
        else:
            total += 1

        if disc_real_img_out.item() > 0.5:
            correct += 1
            total += 1
        else:
            total += 1
    gen_loss_list.append(np.average(temp_gen_loss_list))
    disc_loss_list.append(np.average(temp_disc_loss_list))
    disc_acc_list.append(correct/total)
    plot_and_save(val_image_noise)