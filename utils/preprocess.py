import os 
from glob import glob 
import time 
import cv2 
import numpy as np 

THRESH = 512

images_dir = f"../../../Datasets/AbstractArt"
out_dir = './Clean_Abstract_Art'
os.system(f"mkdir {out_dir}")
all_image_paths = sorted(glob(f"{images_dir}/*.jpg"))
print(f"Total no. of images: {len(all_image_paths)}")

count, n_exceptions = 0, 0
for idx, image_path in enumerate(all_image_paths):
    try:
        image = cv2.imread(image_path)
        height, width, channels = image.shape
        if height >= THRESH and width >= THRESH:
            out_file_path = f"{out_dir}/abs_art_{count}.jpg"
            # center crop the image to (THRESH, THRESH)
            if width > height:
                x_range = [(width-height)//2, (width+height)//2]
                y_range = [0, height]
                cropped = image[y_range[0]:y_range[1], x_range[0]:x_range[1], :]
                resized = cv2.resize(cropped, (THRESH, THRESH))
                cv2.imwrite(out_file_path, resized)
                count += 1
            elif height > width:
                x_range = [0, width]
                y_range = [(height-width)//2, (height+width)//2]
                cropped = image[y_range[0]:y_range[1], x_range[0]:x_range[1], :]
                resized = cv2.resize(cropped, (THRESH, THRESH))
                cv2.imwrite(out_file_path, resized)
                count += 1
            elif height == width:
                resized = cv2.resize(image, (THRESH, THRESH))
                cv2.imwrite(out_file_path, resized)
                count += 1
            else:
                n_exceptions += 1

    except Exception as E:
        n_exceptions += 1
    print(f"  ({idx+1}/{len(all_image_paths)}) | cleaned: {count} | Errors: {n_exceptions}", end='\r')

print(f"\n\nCleaned images: {count}")
