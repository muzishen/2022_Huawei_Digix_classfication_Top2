import os
import cv2
from PIL import Image
import numpy as np

image_path_0 = "/mnt/rjc/reid_data/lane_line/train/0"
image_path_1 = "/mnt/rjc/reid_data/lane_line/train/1"
image_path_test = "/opt/data/private/huawei/digix_data/test"


def read_path(file_pathname):
    num_3 =0
    num_4 = 0
    ii = 0
    imw = []
    imh = []
    for filename in os.listdir(file_pathname):
        #print('image_num:',ii)
        ii+=1
        img = Image.open(file_pathname+'/'+filename)
        img = np.array(img)
        flag = False
        for w, h in zip(imw, imh):
            if w == img.shape[0] and h == img.shape[1]:
                flag = True
                break
        if flag is False:
            imw.append(img.shape[0])
            imh.append(img.shape[1])
            print(img.shape[0], img.shape[1])
        #print(np.array(img).shape)
        if img.shape[2]==3:
            num_3+=1
            #print('num_3',filename)
        elif img.shape[2]==4:
            num_4+=1
            #print('num_4',num_4)
    return num_3,num_4    

num_3_test,num_4_test = read_path(image_path_test)
print('num_3_test,num_4_test',num_3_test,num_4_test)
# num_3_0,num_4_0 = read_path(image_path_0)
# num_3_1,num_4_1 = read_path(image_path_1)
# print('num_3_0,num_4_0:',num_3_0,num_4_0)
# print('num_3_1,num_4_1:',num_3_1,num_4_1)
# my_path = './result'
# with open(my_path, "w") as ff:
#     ff.write('num_3,num_4\n')
#     ff.write('{},{}\n'.format(num_3_0,num_4_0))
#     ff.write('{},{}\n'.format(num_3_1,num_4_1))
