import pandas as pd
import numpy as np
import os
from skimage import io, transform
from PIL import Image
import cv2
import shutil
import glob
import csv

Image.MAX_IMAGE_PIXELS = 1000000000000000


# image_path = "/opt/data/private/Data/lajiao/train"
# new_path = "/opt/data/private/Data/lajiao/train0706"

# train data
def channel4tochannel3(image_path, save_path):
    img_paths = glob.glob(os.path.join(image_path, '*'))
    img_paths.sort()
    labels = csv.reader(open('/opt/data/private/huawei/digix_data/train_label/train_label.csv'))
    for img_path, label in zip(img_paths, labels):
        if int(label[1]) > 0:
            img_png = Image.open(img_path)
            img_png = img_png.convert("RGB")
            image_name = img_path.replace(image_path.split('/')[-1], save_path.split('/')[-1])
            print(image_name)
            img_png.save(image_name)


# val data
image_path = "/opt/data/private/huawei/digix_data/train"
save_path = "/opt/data/private/huawei/digix_data/train3"
#os.mkdir(save_path)
channel4tochannel3(image_path, save_path)
