
# coding: utf-8

# In[21]:

import os
import pickle
import numpy as np
import time
import cv2
import scipy
from PIL import Image
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers import Conv2D, Conv2DTranspose, Dropout
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
from keras.layers.embeddings import Embedding
import keras.backend as K
from keras.utils import to_categorical
import random as rn
import tensorflow as tf
import sys
K.set_image_dim_ordering('tf')

np.random.seed(1)

rn.seed(12345)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

def denorm_img(img):
    img = (img + 1) * 127.5
    return img.astype(np.uint8) 

generator = load_model('generator.hdf5')

with open("eye.pkl", "rb") as input_file:
    eye_dict = pickle.load(input_file)
with open("hair.pkl", "rb") as input_file:
    hair_dict = pickle.load(input_file)

noise_shape = (1,1,100)
test = open(sys.argv[1],'r')
if not os.path.exists("samples/"):
    os.mkdir("samples/")
for i in test:
    split = i.split(",")
    tags = split[1].split()
    hair = np.full(5, 11)
    eyes = np.full(5, 10)
    if(len(tags) == 4):
            if tags[-1] == "hair":
                hair_type = tags[-2]+" hair"
                eyes_type = tags[0] +" eyes"
            if tags[-1] == 'eyes':
                hair_type = tags[0]+" hair"
                eyes_type = tags[-2]+" eyes"
            hair = np.full(5, hair_dict[hair_type])
            eyes = np.full(5, eye_dict[eyes_type])
    if(len(tags) == 2):
            if tags[-1] == "hair":
                hair_type = tags[0]+" hair"
                hair = np.full(5, hair_dict[hair_type])
            if tags[-1] == 'eyes':
                eyes_type = tags[0]+" eyes"
                eyes = np.full(5, eye_dict[eyes_type])
    noise = np.random.normal(0, 1, size=(5,)+noise_shape)            
    pics = generator.predict([noise, hair, eyes])
    
    import scipy.misc
    for idx, i in enumerate(pics):
        dir_ = 'samples/sample_'+split[0]+'_'+str(idx+1) + '.jpg'
        scipy.misc.toimage(denorm_img(i)).save(dir_)     


