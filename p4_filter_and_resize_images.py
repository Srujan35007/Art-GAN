import time 
from tqdm import tqdm 
import os 
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
from tqdm import tqdm
print('Imports complete')

path = f'Path_to_the_downloaded_images_folder'
resized = f'Path_to_the_new_resized_images_folder'

file_paths = []
for roots, dirs, files in os.walk(path):
    for file_ in files:
        file_paths.append(f'{path}\\{file_}')

print('Total file_paths = ', len(file_paths))
count = 0
time.sleep(2)
for file_path in tqdm(file_paths):
    try:
        img = cv2.imread(file_path)
        if img.shape[0] > img.shape[1]: # Height > Width
            width = img.shape[1]
            height = img.shape[0]
            # If a potrait image crop mostly to the upper half of the image
            # cz' that's where most of the potrait art details are
            new_height_start = max(0, int((height-width)//2 - 0.4*(height-width)//2))
            new_height_end = new_height_start + width
            img = img[new_height_start:new_height_end, 0:width]
            img = cv2.resize(img, (400,400))
            cv2.imwrite(resized+f'\\art_400_{count}.jpg', img)
            count += 1
        elif img.shape[0] < img.shape[1]: # Heigth < Width
            width = img.shape[1]
            height = img.shape[0]
            # It it's a landscape image just go for center crop
            new_width_start = max(0, int((width-height)//2))
            new_width_end = new_width_start + height
            img = img[0:height, new_width_start:new_width_end]
            img = cv2.resize(img, (400,400))
            cv2.imwrite(resized+f'\\art_400_{count}.jpg', img)
            count += 1
        elif img.shape[0] == img.shape[1]: # Square image
            img = cv2.resize(img, (400,400))
            cv2.imwrite(resized+f'\\art_400_{count}.jpg', img)
            count += 1
        else:
            pass
    except:
        pass
print(f'Total valid images = {count}')
