import pandas as pd
import numpy as np
import os
from skimage import io, transform
import shutil
import csv

base="./results_csv/submission_90.8301.csv"     #91.5976
small_1280 = "./results_csv/epoch30_submission_91.1579.csv"  #91.2559
ibna101="./results_csv/submission_epoch30_89.80.csv"     #91.4493
tiny= "./results_csv/submission_epoch30_90.7260.csv"  # 91.4846
swin_small =  "./results_csv/submission_epoch30_90.2331.csv" #91.5578

# next_ibn =  "/opt/data/private/Logs/classfication/digix/jx_next101v2/submission_epoch30_88.8660.csv"  #91.4903

new_csv = "submission.csv"
def zc_read_csv(path):
    zc_dataframe = pd.read_csv(path, sep=",") 
    x = []
    y = []
    for zc_index in zc_dataframe.index:
        zc_line = zc_dataframe.loc[zc_index]
        x.append(zc_line["imagename"])
        y.append(zc_line["defect_prob"])
    return  x, y

name1,prob1= zc_read_csv(base)
_,prob2= zc_read_csv(small_1280)
_,prob3= zc_read_csv(ibna101)
_,prob4= zc_read_csv(tiny)
_,prob5= zc_read_csv(swin_small)


with open(new_csv, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["imagename", "defect_prob"])
        for i in range(len(name1)):
            template = [name1[i], prob1[i]+ prob2[i] + prob3[i]+ prob4[i] + prob5[i]]
            writer.writerow(template)
csvfile.close()