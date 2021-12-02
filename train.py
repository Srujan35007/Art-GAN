import time 
import os 
from datetime import datetime
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ArtData
from models import Generator, Discriminator


# Hyperparameters and config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
LEARNING_RATE = 0.0003
BETA1 = 0.5
BETA2 = 0.99
loss_fn = nn.BCELoss()
REAL_LABEL = 1.0
FAKE_LABEL = 0.0

N_EPOCHS = 100
BATCH_SIZE = 8
TRAINING_SAMPLES_LIMIT = 1024


# Some helper functions
get_timestamp = lambda : datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

def append_to_logfile(file_path, content):
    with open(file_path, 'a') as write_file:
        write_file.write(content)


# Load dataset
DATASET_PATH = f""
train_transforms = transforms.Compose(
    transforms.ToTensor(),
    transforms.RandomRotation(30),
)
train_data = ArtData(DATASET_PATH, limit=TRAINING_SAMPLES_LIMIT, transform=train_transforms)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)


# Initialize models and optimizers
netG = Generator().to(device)
netD = Discriminator().to(device)

# metrics
g_losses_per_epoch = []
d_losses_per_epoch = []
logs_path = f'./logs/GAN_RUN_{get_timestamp()}'
os.system(f'mkdir -p {logs_path}')
metrics_logfile_path = f'{logs_path}/metrics.csv'
fixed_noise = torch.randn(16,16,16)

