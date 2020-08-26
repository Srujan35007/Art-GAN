import time 
print('Imports started')
b = time.time()
import os 
from pathlib import Path 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf 
from tensorflow.keras import models, layers, Model, Input
import numpy as np 
import matplotlib.pyplot as plt 
a = time.time()
print('Imports complete in {0:.2f} seconds'.format(a-b))


# Generator
def make_generator(latent_dims = 200, max_channels=256):
    n = max_channels
    act = layers.LeakyReLU(1)

    input_ = Input(shape=(latent_dims,))
    lin1 = act(layers.Dense(625)(input_))
    reshape = layers.Reshape((25,25,1))(lin1)
    x = layers.Conv2DTranspose(n,    3,1,'same')(reshape)
    x = layers.Conv2DTranspose(n,    3,1,'same')(x)
    x = layers.Conv2DTranspose(n,    3,1,'same')(x)
    x = layers.Conv2DTranspose(n,    4,2,'same')(x) # 50,50
    x = layers.Conv2DTranspose(n//2, 3,1,'same')(x)
    x = layers.Conv2DTranspose(n//2, 3,1,'same')(x)
    x = layers.Conv2DTranspose(n//2, 3,1,'same')(x)
    x = layers.Conv2DTranspose(n//2, 4,2,'same')(x) # 100,100
    x = layers.Conv2DTranspose(n//4, 3,1,'same')(x)
    x = layers.Conv2DTranspose(n//4, 3,1,'same')(x)
    x = layers.Conv2DTranspose(n//4, 3,1,'same')(x)
    x = layers.Conv2DTranspose(n//4, 4,2,'same')(x) # 200,200
    x = layers.Conv2DTranspose(n//8, 3,1,'same')(x)
    x = layers.Conv2DTranspose(n//8, 3,1,'same')(x)
    x = layers.Conv2DTranspose(n//8, 3,1,'same')(x)
    x = layers.Conv2DTranspose(n//8, 4,2,'same')(x) # 400,400
    output_ = layers.Activation('sigmoid')(layers.Conv2DTranspose(3,    3,1,'same')(x))

    model = Model(input_, output_)
    return model

gen = make_generator()
a = np.random.random((1,200))
plt.imshow(gen.predict(a).reshape(400,400,3))
plt.show()