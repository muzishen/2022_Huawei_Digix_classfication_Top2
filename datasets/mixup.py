import csv
import shutil
import skimage.io as io
from skimage.transform import resize
import os
import random
import numpy as np
import copy
import glob
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000000000

def channel4tochannel3(img_path):
    img_png = Image.open(img_path)
    img_png = img_png.convert("RGB")
    return img_png

def Random_TransMix(img_list, height, width, ratio):
    image_num = len(img_list)
    per_image_wdith = width
    per_image_height = height // image_num
    for i in range(image_num):
        img = img_list[i]
        height_ratio = random.uniform(ratio[0], ratio[1])
        width_ratio = random.uniform(ratio[0], ratio[1])
        height1, width1 = img.shape[:2]
        crop_height = int(height1 * height_ratio)
        crop_width = int(width1 * width_ratio)
        y1 = (height1 - crop_height) // 2
        y2 = y1 + crop_height
        x1 = (width1 - crop_width) // 2
        x2 = x1 + crop_width
        img = img[y1:y2, x1:x2]
        img = resize(img, (per_image_height, per_image_wdith), preserve_range=True)
        if i == 0:
            concate_img = img
        else:
            print(concate_img.shape, img.shape)
            concate_img = np.concatenate((concate_img, img), axis=0)
        concate_img = concate_img.astype(np.uint8)

    return concate_img


def chocie_image(total_list, num, height, width, ratio):
    concate_img_num = num
    total_img_list = copy.deepcopy(total_list)
    label = 0
    for class_img_list in total_img_list:
        label += 1
        while len(class_img_list) >= concate_img_num:
            # 从单个类别图像序列中获取图片列表
            img_list = random.sample(class_img_list, concate_img_num)
            concate_img = []
            for img in img_list:
                class_img_list.remove(img)
                img_path = os.path.join(train_path, img)
                load_img = io.imread(img_path)
                concate_img.append(load_img)
            # 形成新的拼接图象
            concate_img = Random_TransMix(concate_img, height, width, ratio)
            for img in img_list:
                nn = img.split('/')[-1]
                fold_name = os.path.join(save_path, str(label))
                if not os.path.exists(fold_name):
                    os.makedirs(fold_name)
                save_img_name = str(label) + '/_con' + str(concate_img_num) + '-' + nn
                print(save_img_name)
                save_img_path = os.path.join(save_path, save_img_name)
                io.imsave(save_img_path, concate_img)


def get_root(img_paths):
    total_list = []
    image0_list = []
    image1_list = []
    img_paths = glob.glob(os.path.join(img_paths, '*'))
    img_paths.sort()
    labels = csv.reader(open('/opt/data/private/huawei/digix_data/train_label/train_label.csv'))
    for img_path in img_paths:
        image1_list.append(img_path)
        #else:
        #    image0_list.append(img_path)
    #total_list.append(image0_list)
    total_list.append(image1_list)
    # for root, dirs, files in os.walk(image_path):
    #     for dir in dirs:
    #         dir_path = os.path.join(root, dir)
    #         img_list = glob.glob(dir_path + '/*')
    #         total_list.append(img_list)
    return total_list


if __name__ == '__main__':
    train_path = '/opt/data/private/huawei/digix_data/train3'
    save_path = '/opt/data/private/huawei/digix_data/train_mix'
    total_list = get_root(train_path)
    print(len(total_list))
    for num in range(2, 5):
        chocie_image(total_list, num, height=2400, width=1080, ratio=[1, 1])
