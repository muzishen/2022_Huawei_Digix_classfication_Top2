# -------------------------
# Author: xin jiang
# @Time : 2022/8/21 21:11
# -------------------------
from skimage import io
import os

from PIL import Image
import numpy as np

image_path = '/opt/data/private/huawei/digix_data/test'
save_path = '/opt/data/private/huawei/digix_data/crop_test'


def crop_image(path):
    for filename in os.listdir(path):
        print("Cropping : {}".format(filename))
        img = Image.open(path + '/' + filename)
        # img = Image.open('C:/Users/xjiang/Desktop/train_10495.png')
        img = np.array(img)
        # if img.shape[0] == 2400:
        #     img = img[300:-420, :, :]
        # elif img.shape[0] == 2340:
        #     img = img[260:-300, :, :]
        # elif img.shape[0] == 1560:
        #     img = img[70:-160, :, :]
        if img.shape[0] == 2400:
            img = img[120:-240, :, :]
        elif img.shape[0] == 2340:
            img = img[70:-120, :, :]
        elif img.shape[0] == 1560:
            img = img[70:-160, :, :]
        img = img.astype(np.uint8)
        save_img = save_path + '/' + filename[:-4] + '.png'
        io.imsave(save_img, img)


if __name__ == '__main__':
    crop_image(image_path)
