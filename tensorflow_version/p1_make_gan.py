import time 
print('Imports started')
b = time.time()
import os 
from pathlib import Path 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf 
from tensorflow.keras import models, layers, Model, Input
a = time.time()
print('Imports complete in {0:.2f} seconds'.format(a-b))

# Generator
input_ = Input(shape=(25,25,1))
conv1 = layers.Conv2DTranspose(20, 3,1,'same')(input_)
conv2 = layers.Conv2DTranspose(40, 4,2,'same')(conv1)
conv3 = layers.Conv2DTranspose(50, 3,1,'same')(conv2)
conv4 = layers.Conv2DTranspose(100, 4,2,'same')(conv3)
output_ = layers.Conv2DTranspose(3, 4,2,'same')(conv4)
model = Model(input_, output_)
