import pandas as pd
from scipy import misc
import cv2
import numpy as np
import sys
import os


data_entry = sys.argv[1]
train_txt = sys.argv[2]
test_txt = sys.argv[3]

LABELS = {'Atelectasis':0,'Cardiomegaly':1,'Effusion':2,'Infiltration':3,'Mass':4,'Nodule':5, 'Pneumonia':6,'Pneumothorax':7}
data = open(data_entry,'r')
training_idx = open(train_txt,'r')
training_set = {}
for i in training_idx:
    training_set[i[:-1]] = 1

Imgs = []
Ages = []
Genders = []
Positions = []
Labels = []
count = 0
for i in data:
    if count == 0:
        count = 1
        continue
    split = i.split(',')
    labels = split[1].split("|")
    if split[0] in training_set:
        label = np.zeros(8)
        for v in labels:
            if v in LABELS:
                label[LABELS[v]]=1 
        if np.sum(label) != 0:
            Imgs.append(split[0])
            Ages.append(split[4])
            Genders.append(split[5])
            Positions.append(split[6])
            Labels.append(label)
output = open("train_multi.csv",'w')
output.write("Image,Age,Gener,Position,Atelectasis,Cardiomegaly,Effusion,Infiltration,Mass,Nodule,Pneumonia,Pneumothorax\n")
for img, age, gender, pos, label in zip(Imgs,Ages,Genders,Positions,Labels):
    output.write(img+","+age+","+gender+","+pos+",")
    output.write(str(label[0]))
    for i in label[1:]:
        output.write(","+str(i))
    output.write("\n")
output.close()

import pickle
with open('label_dict.pickle', 'wb') as handle:
    pickle.dump(LABELS, handle, protocol=pickle.HIGHEST_PROTOCOL)

LABELS = {'Atelectasis':0,'Cardiomegaly':1,'Effusion':2,'Infiltration':3,'Mass':4,'Nodule':5, 'Pneumonia':6,'Pneumothorax':7}
data = open(data_entry,'r')
testing_idx = open(test_txt,'r')
testing_set = {}
for i in testing_idx:
    testing_set[i[:-1]] = 1

Imgs = []
Ages = []
Genders = []
Positions = []
Labels = []
count = 0
for i in data:
    if count == 0:
        count = 1
        continue
    split = i.split(',')
    labels = split[1].split("|")
    if split[0] in testing_set:
        label = np.zeros(8)
        for v in labels:
            if v in LABELS:
                label[LABELS[v]]=1
        if np.sum(label) != 0:
            Imgs.append(split[0])
            Ages.append(split[4])
            Genders.append(split[5])
            Positions.append(split[6])
            Labels.append(label)

output = open("valid_multi.csv",'w')
output.write("Image,Age,Gener,Position,Atelectasis,Cardiomegaly,Effusion,Infiltration,Mass,Nodule,Pneumonia,Pneumothorax\n")
for img, age, gender, pos, label in zip(Imgs,Ages,Genders,Positions,Labels):
    output.write(img+","+age+","+gender+","+pos+",")
    output.write(str(label[0]))
    for i in label[1:]:
        output.write(","+str(i))
    output.write("\n")
output.close()

