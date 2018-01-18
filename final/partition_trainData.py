import pandas as pd
from scipy import misc
import cv2
import numpy as np
import sys
import os
import pickle

image_dir = sys.argv[1]
data = pd.read_csv("train_multi.csv")
data = pd.get_dummies(data,columns=['Gener','Position'])

for _i in range(6):
    print(_i)
    imgs = []
    for i in list(data['Image'])[_i*7500:(_i+1)*7500]:
        img = misc.imread(os.path.join(image_dir,i))
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img = cv2.equalizeHist(img)
        img = misc.imresize(img, (256, 256))
        imgs.append(img)
    imgs = np.asarray(imgs)
    np.save('train_multi_'+str(_i)+'.npy', imgs)

data = pd.read_csv("valid_multi.csv")
data = pd.get_dummies(data,columns=['Gener','Position'])

imgs = []
for i in list(data['Image']):
    img = misc.imread(os.path.join(image_dir,i))
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = cv2.equalizeHist(img)
    img = misc.imresize(img, (256, 256))
    imgs.append(img)
imgs = np.asarray(imgs)
np.save('valid_multi.npy', imgs)

