import time 
import os 
from datetime import datetime
import numpy as np 
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from dataset import ArtData
from models import Generator, Discriminator


# Hyperparameters and config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
Z_INPUT_SHAPE = [4,16,16]
Z_DIMS = np.prod(Z_INPUT_SHAPE)
D_OUT_SHAPE = [256,16,16]
LEARNING_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.999
loss_fn = nn.BCELoss()
REAL_LABEL = 1
FAKE_LABEL = 0

N_EPOCHS = 100
BATCH_SIZE = 8
TRAINING_SAMPLES_LIMIT = 1024


# Some helper functions
get_timestamp = lambda : datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

def append_to_logfile(file_path, content):
    with open(file_path, 'a') as write_file:
        write_file.write(content)


# Load dataset
DATASET_PATH = f"../../Datasets/Clean_Abstract_Art"
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    #transforms.RandomRotation(30),
    ])
train_data = ArtData(DATASET_PATH, limit=TRAINING_SAMPLES_LIMIT, transform=train_transforms)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
print(f"Data loaded")


# Initialize models and optimizers
netG = Generator().to(device)
netD = Discriminator().to(device)
optimG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=[BETA1, BETA2])
optimD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=[BETA1, BETA2])

# metrics
g_losses_per_epoch = []
d_losses_per_epoch = []
logs_path = f'./logs/GAN_RUN_{get_timestamp()}'
os.system(f'mkdir -p {logs_path}')
test_images_dir = f"{logs_path}/generated_images"
os.system(f"mkdir -p {test_images_dir}")
save_models_dir = f"{logs_path}/saved_models"
os.system(f"mkdir -p {save_models_dir}")
metrics_logfile_path = f'{logs_path}/metrics.csv'
logfile_column_headers = "EPOCHS,BATCHES,LOSSES_G,LOSSES_D\n"
append_to_logfile(metrics_logfile_path, logfile_column_headers)
fixed_noise = torch.randn(16, Z_DIMS)

# Start training
print(f"Started training on device: {device}\n\n")
start_clock = time.perf_counter()
for epoch_idx in range(N_EPOCHS):
    # For batch metrics
    per_batch_log = ''
    g_losses_per_batch = []
    d_losses_per_batch = []
    for batch_idx, real in enumerate(train_loader):
        _batch_size = real.shape[0]
        '''Train Discriminator:
        - Maximize: log(D(real)) + log(1 - D(G(z)))
        '''
        netD.zero_grad()
        # Train on real samples
        real = real.to(device)
        output = netD(real).view(_batch_size, *D_OUT_SHAPE) 
        label = torch.rand((_batch_size, *D_OUT_SHAPE), dtype=torch.float, device=device)
        label = label.fill_(REAL_LABEL)
        lossD_real = loss_fn(output, label)
        lossD_real.backward()
        # Train on fake samples
        noise = torch.randn((_batch_size, Z_DIMS,), dtype=torch.float, device=device)
        fake = netG(noise.view(_batch_size, *Z_INPUT_SHAPE))
        output = netD(fake.detach()).view(_batch_size, *D_OUT_SHAPE)
        label.fill_(FAKE_LABEL)
        lossD_fake = loss_fn(output, label)
        lossD_fake.backward()
        # optimD step
        lossD = lossD_fake + lossD_real
        optimD.step()
        '''Train Generator:
        Maximize: log(D(G(z)))
        '''
        netG.zero_grad()
        output = netD(fake).view(_batch_size, *D_OUT_SHAPE)
        label.fill_(REAL_LABEL)
        lossG = loss_fn(output, label)
        lossG.backward()
        optimG.step()
        # Batch metrics
        per_batch_log = per_batch_log + \
                        f"{epoch_idx+1},{batch_idx+1},{lossG.item()},{lossD.item()}\n"
        g_losses_per_batch.append(lossG.item())
        d_losses_per_batch.append(lossD.item())
        # Display batch metrics
        if batch_idx % (len(train_loader)//10) == 0:
            print(f"\tEpoch({epoch_idx+1}/{N_EPOCHS}) |", end=' ') 
            print(f"Batch: ({batch_idx+1}/{len(train_loader)}) |", end=' ')
            print(f"LossG: {np.average(g_losses_per_batch):.6f} |", end=' ')
            print(f"LossD: {np.average(d_losses_per_batch):.6f}")
    
    # Epoch metrics
    g_losses_per_epoch.append(np.average(g_losses_per_batch))
    d_losses_per_epoch.append(np.average(d_losses_per_batch))
    append_to_logfile(metrics_logfile_path, per_batch_log)
    # Generate and save test images
    gen_images = []
    for noise in fixed_noise:
        noise = noise.view(1, *Z_INPUT_SHAPE).to(device)
        out = netG(noise).cpu()
        gen_images.append(out[0])
    grid = torchvision.utils.make_grid(gen_images, nrow=4, padding=8)
    torchvision.utils.save_image(grid, f"{test_images_dir}/epoch_{epoch_idx+1}.jpg")
    # Display epoch metrics
    end_epoch_clock = time.perf_counter()
    elapsed = end_epoch_clock-start_clock
    print(f"Epoch: {epoch_idx+1} | Elapsed: {elapsed/60:.1f} Min. |", end=' ')
    print(f"LossG: {g_losses_per_epoch[-1]:.8f} | LossD: {d_losses_per_epoch[-1]:.8f}\n")
