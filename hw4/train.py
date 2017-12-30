
# coding: utf-8

# In[ ]:


"""
Created on Sat Jul 15 12:41:47 2017
@author: Pavitrakumar
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import os
import numpy as np
from sklearn.utils import shuffle
import time
import cv2
import tqdm
from PIL import Image
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import GaussianNoise
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten, Dropout
from keras.layers import Input, merge, concatenate, multiply
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt
#from misc_layers import MinibatchDiscrimination, SubPixelUpscaling, CustomLRELU, bilinear2x
#from keras_contrib.layers.convolutional import SubPixelUpscaling
from keras.datasets import mnist
import keras.backend as K
import tensorflow as tf
import random as rn
from keras.initializers import RandomNormal
K.set_image_dim_ordering('tf')


# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)



def get_gen_normal(noise_shape):
    noise_shape = noise_shape
    """
    Changing padding = 'same' in the first layer makes a lot fo difference!!!!
    """
    #kernel_init = RandomNormal(mean=0.0, stddev=0.01)
    kernel_init = 'glorot_uniform'
    
    gen_input = Input(shape = noise_shape) #if want to directly use with conv layer next
    
    hair_input = Input(shape = (1,))
    hair = Embedding(hair_len, 8)(hair_input)
    hair_reshape = Reshape((1,1,8), input_shape=(1,8))(hair)

    eyes_input = Input(shape = (1,))
    eyes = Embedding(eyes_len, 8)(eyes_input)
    eyes_reshape = Reshape((1,1,8), input_shape=(1,8))(eyes)
    
    concat = concatenate([gen_input, hair_reshape, eyes_reshape])
    
    generator = Conv2DTranspose(filters = 512, kernel_size = (4,4), strides = (1,1), padding = "valid", kernel_initializer = kernel_init)(concat)
    generator = BatchNormalization(momentum = 0.5)(generator)
    generator = LeakyReLU(0.2)(generator)

    generator = Conv2DTranspose(filters = 256, kernel_size = (4,4), strides = (2,2), padding = "same", kernel_initializer = kernel_init)(generator)
    generator = BatchNormalization(momentum = 0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
    
    generator = Conv2DTranspose(filters = 128, kernel_size = (4,4), strides = (2,2), padding = "same", kernel_initializer = kernel_init)(generator)
    generator = BatchNormalization(momentum = 0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
        
    generator = Conv2DTranspose(filters = 64, kernel_size = (4,4), strides = (2,2), padding = "same", kernel_initializer = kernel_init)(generator)
    generator = BatchNormalization(momentum = 0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
    
    generator = Conv2DTranspose(filters = 64, kernel_size = (3,3), strides = (1,1), padding = "same", kernel_initializer = kernel_init)(generator)
    generator = BatchNormalization(momentum = 0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
    
    generator = Conv2DTranspose(filters = 3, kernel_size = (4,4), strides = (2,2), padding = "same",  kernel_initializer = kernel_init)(generator)
    generator = Activation('tanh')(generator)
        
    gen_opt = Adam(lr=0.0002, beta_1=0.5)
    generator_model = Model(input = [gen_input, hair_input, eyes_input], output = generator)
    generator_model.compile(loss='binary_crossentropy', optimizer=gen_opt, metrics=['accuracy'])
    generator_model.summary()

    return generator_model
    
#------------------------------------------------------------------------------------------

def get_disc_normal(image_shape=(64,64,3)):

    kernel_init = 'glorot_uniform'
    
    dis_input = Input(shape = image_shape)
    
    discriminator = Conv2D(filters = 64, kernel_size = (3,3), strides = (2,2), padding = "same", kernel_initializer = kernel_init)(dis_input)
    discriminator = LeakyReLU(0.2)(discriminator)

    discriminator = Conv2D(filters = 128, kernel_size = (3,3), strides = (2,2), padding = "same", kernel_initializer = kernel_init)(discriminator)
    discriminator = BatchNormalization(momentum = 0.5)(discriminator)
    discriminator = LeakyReLU(0.2)(discriminator)

    discriminator = Conv2D(filters = 256, kernel_size = (3,3), strides = (2,2), padding = "same", kernel_initializer = kernel_init)(discriminator)
    discriminator = BatchNormalization(momentum = 0.5)(discriminator)
    discriminator = LeakyReLU(0.2)(discriminator)

    discriminator = Conv2D(filters = 512, kernel_size = (3,3), strides = (2,2), padding = "same", kernel_initializer = kernel_init)(discriminator)
    discriminator = BatchNormalization(momentum = 0.5)(discriminator)
    discriminator = LeakyReLU(0.2)(discriminator)
    
    discriminator = Flatten()(discriminator)

    img = Dense(1)(discriminator)
    img = Activation('sigmoid')(img)
    
    hair_output = Dense(hair_len)(discriminator)
    hair_output = Activation('softmax')(hair_output)
    
    eyes_output = Dense(eyes_len)(discriminator)
    eyes_output = Activation('softmax')(eyes_output)
    
    dis_opt = Adam(lr=0.0002, beta_1=0.5)
    discriminator_model = Model(input = dis_input, output = [img, hair_output, eyes_output])
    discriminator_model.compile(loss=['binary_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
                                loss_weights=[0.5,0.25,0.25],
                                optimizer=dis_opt, 
                                metrics=['accuracy'])
    discriminator_model.summary()
    return discriminator_model

#------------------------------------------------------------------------------------------

import os
import glob
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from sklearn.utils import shuffle
import time
import cv2
import scipy
import imageio
from PIL import Image
import matplotlib.gridspec as gridspec
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
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.layers.embeddings import Embedding
import keras.backend as K
from scipy.interpolate import spline
from keras.utils import to_categorical
from numpy import linalg as LA
K.set_image_dim_ordering('tf')

from collections import deque

def norm_img(img):
    img = (img / 127.5) - 1
    return img

def denorm_img(img):
    img = (img + 1) * 127.5
    return img.astype(np.uint8) 


def sample_from_dataset(batch_size, image_shape, data_dir=None, data = None):
    sample_dim = (batch_size,) + image_shape
#     hair_dim = (batch_size, hair_len)
#     eyes_dim = (batch_size, eyes_len)
    hair_dim = (batch_size, 1)
    eyes_dim = (batch_size, 1)
    sample_img = np.empty(sample_dim, dtype=np.float32)
    sample_hair = np.empty(hair_dim, dtype=np.float32)
    sample_eyes = np.empty(eyes_dim, dtype=np.float32)
    all_data_dirlist = list(glob.glob(data_dir))
    rand_ids = np.random.choice(ids,batch_size)
    for index,Id in enumerate(rand_ids):
        image = Image.open('faces/'+str(Id)+'.jpg')
        #print(image.size)
        #image.thumbnail(image_shape[:-1], Image.ANTIALIAS) - this maintains aspect ratio ; we dont want that - we need m x m size
        image = image.resize(image_shape[:-1])
        image = image.convert('RGB') #remove transparent ('A') layer
        #print(image.size)
        #print('\n')
        image = np.asarray(image)
        image = norm_img(image)
        sample_img[index,...] = image
#         sample_hair[index,...] = hair[Id]
#         sample_eyes[index,...] = eyes[Id]
        sample_hair[index,...] = np.argmax(np.asarray(hair[Id])) 
        sample_eyes[index,...] = np.argmax(np.asarray(eyes[Id]))
    return sample_img, sample_hair, sample_eyes 


def gen_noise(batch_size, noise_shape):
    rand_hair = np.random.randint(hair_len, size = batch_size) 
    rand_eyes = np.random.randint(eyes_len, size = batch_size) 
    return np.random.normal(0, 1, size=(batch_size,)+noise_shape), rand_hair, rand_eyes

def generate_images(generator, save_dir):
    noise = gen_noise(batch_size,noise_shape)
    fake_data_X = generator.predict(noise)
    print("Displaying generated images")
    plt.figure(figsize=(4,4))
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0, hspace=0)
    rand_indices = np.random.choice(fake_data_X.shape[0],16,replace=False)
    for i in range(16):
        #plt.subplot(4, 4, i+1)
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        rand_index = rand_indices[i]
        image = fake_data_X[rand_index, :,:,:]
        fig = plt.imshow(denorm_img(image))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir+str(time.time())+"_GENERATEDimage.png",bbox_inches='tight',pad_inches=0)
    plt.show()


def save_img_batch(img_batch,img_save_dir):
    plt.figure(figsize=(4,4))
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0, hspace=0)
    rand_indices = np.random.choice(img_batch.shape[0],16,replace=False)
    for i in range(16):
        #plt.subplot(4, 4, i+1)
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        rand_index = rand_indices[i]
        image = img_batch[rand_index, :,:,:]
        fig = plt.imshow(denorm_img(image))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(img_save_dir,bbox_inches='tight',pad_inches=0)
    plt.close('all')
    #plt.show()   

def d_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

noise_shape = (1,1,100)
num_steps = 50000
batch_size = 128

image_shape = None

img_save_dir = "test_output/"

save_model = False



# image_shape = (96,96,3)
image_shape = (64,64,3)
data_dir =  "faces/*.jpg"
csvfile = open('preprocess.csv','r')
ids = []
hair = {}
eyes = {}
hair_len = 12
eyes_len = 11
for row in csvfile:
    split = row.split(',')
    Id = split[0]
    ids.append(Id)
    eyes[Id] = split[1:12]
    hair[Id] = split[12:]
#data_dir = "E:\\GAN_Datasets\\curl\\online_ds\\thumb\\*\\*.png"

log_dir = img_save_dir
save_model_dir = img_save_dir

discriminator = get_disc_normal(image_shape)
generator = get_gen_normal(noise_shape)

#generator = load_model(save_model_dir+'9999_GENERATOR_weights_and_arch.hdf5')
#discriminator = load_model(save_model_dir+'9999_DISCRIMINATOR_weights_and_arch.hdf5')

discriminator.trainable = False

opt = Adam(lr=0.0002, beta_1=0.5) #same as gen

gen_inp = Input(shape = noise_shape)
# hair_inp = Input(shape = (1, 1, hair_len))
# eyes_inp = Input(shape = (1, 1, eyes_len))
hair_inp = Input(shape = (1,))
eyes_inp = Input(shape = (1,))

GAN_inp = generator([gen_inp,hair_inp,eyes_inp])

GAN_opt = discriminator(GAN_inp)
gan = Model(input = [gen_inp, hair_inp, eyes_inp] , output = GAN_opt)
gan.compile(loss = 'binary_crossentropy', optimizer = opt, metrics=['accuracy'])
gan.summary()


avg_disc_fake_loss = deque([0], maxlen=250)     
avg_disc_real_loss = deque([0], maxlen=250)
avg_GAN_loss = deque([0], maxlen=250)

OBSERB = open("observ.csv",'w')


for step in range(num_steps): 
    tot_step = step
    print("Begin step: ", tot_step)
    step_begin_time = time.time() 
    
    real_data_img, real_data_hair, real_data_eyes = sample_from_dataset(batch_size, image_shape, data_dir = data_dir)
#     real_data_hair_reshape = real_data_hair.reshape((len(real_data_hair),1,1,len(real_data_hair[0])))
#     real_data_eyes_reshape = real_data_eyes.reshape((len(real_data_eyes),1,1,len(real_data_eyes[0])))
    real_data_hair_reshape = to_categorical(real_data_hair, num_classes=hair_len) #+ np.random.random_sample(size=(batch_size,hair_len))*0.2
    real_data_eyes_reshape = to_categorical(real_data_eyes, num_classes=eyes_len) #+ np.random.random_sample(size=(batch_size,eyes_len))*0.2
    #real_data_hair_reshape /= LA.norm(real_data_hair_reshape, axis=1)[:,None]
    #real_data_eyes_reshape /= LA.norm(real_data_eyes_reshape, axis=1)[:,None]

    noise, noise_hair, noise_eyes = gen_noise(batch_size,noise_shape)
#     noise_hair_reshape = noise_hair.reshape((len(noise_hair),1,1,len(noise_hair[0])))
#     noise_eyes_reshape = noise_eyes.reshape((len(noise_eyes),1,1,len(noise_eyes[0])))
    noise_hair_reshape = to_categorical(noise_hair, num_classes=hair_len) #+ np.random.random_sample(size=(batch_size,hair_len))*0.2
    noise_eyes_reshape = to_categorical(noise_eyes, num_classes=eyes_len) #+ np.random.random_sample(size=(batch_size,eyes_len))*0.2
    #noise_hair_reshape /= LA.norm(noise_hair_reshape, axis=1)[:,None]
    #noise_eyes_reshape /= LA.norm(noise_eyes_reshape, axis=1)[:,None]
    
#     fake_data_img = generator.predict([noise, noise_hair_reshape, noise_eyes_reshape])
    fake_data_img = generator.predict([noise, noise_hair, noise_eyes])
    
    if (tot_step % 50) == 0:
        step_num = str(tot_step).zfill(4)
        save_img_batch(fake_data_img,img_save_dir+step_num+"_image.png")

    data_X = np.concatenate([real_data_img,fake_data_img])

    soft = np.random.random_sample(batch_size)
    real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
    fake_data_Y = np.random.random_sample(batch_size)*0.2
#     real_data_Y = np.ones(batch_size)

#     fake_data_Y = np.zeros(batch_size)
    #fake_data_Y = np.zeros(batch_size)
     
    data_Y = np.concatenate((real_data_Y,fake_data_Y))
    
        
    discriminator.trainable = True
    generator.trainable = False
    
#     dis_metrics_real = discriminator.train_on_batch(real_data_img,[real_data_Y, real_data_hair, real_data_eyes])   #training seperately on real
#     dis_metrics_fake = discriminator.train_on_batch(fake_data_img,[fake_data_Y, noise_hair, noise_eyes])   #training seperately on fake
    if step%2 == 0:
        dis_metrics_real = discriminator.fit(real_data_img,[real_data_Y, real_data_hair_reshape, real_data_eyes_reshape],epochs=1, batch_size=batch_size,verbose=0)   #training seperately on real
        dis_metrics_fake = discriminator.fit(fake_data_img,[fake_data_Y, noise_hair_reshape, noise_eyes_reshape],epochs=1, batch_size=batch_size,verbose=0)   #training seperately on fake 
   
    
    
    avg_disc_fake_loss.append(np.mean(dis_metrics_fake.history['loss']))
    avg_disc_real_loss.append(np.mean(dis_metrics_real.history['loss']))
    
    generator.trainable = True

    GAN_img, GAN_hair, GAN_eyes = gen_noise(batch_size,noise_shape)
    GAN_hair_reshape = to_categorical(GAN_hair, num_classes=hair_len) #+ np.random.random_sample(size=(batch_size,hair_len))*0.2
    GAN_eyes_reshape = to_categorical(GAN_eyes, num_classes=eyes_len) #+ np.random.random_sample(size=(batch_size,eyes_len))*0.2

#     GAN_Y = [real_data_Y, real_data_hair, real_data_eyes]
    GAN_Y = [real_data_Y, GAN_hair_reshape, GAN_eyes_reshape]
    
    discriminator.trainable = False
    
#     gan_metrics = gan.train_on_batch([GAN_img, 
#                                       GAN_hair.reshape(len(GAN_hair),1,1,len(GAN_hair[0])), 
#                                       GAN_eyes.reshape(len(GAN_eyes),1,1,len(GAN_eyes[0]))], GAN_Y)
    gan_metrics = gan.fit([GAN_img, GAN_hair, GAN_eyes], GAN_Y, epochs=1, batch_size=batch_size, verbose=0)
    OBSERB.write("%f, %f, %f\n"% (np.mean(dis_metrics_real.history['loss']), np.mean(dis_metrics_fake.history['loss']),np.mean(gan_metrics.history['loss'])))
    print("Disc: real loss: %f fake loss: %f GAN loss: %f" % (np.mean(dis_metrics_real.history['loss']), np.mean(dis_metrics_fake.history['loss']),np.mean(gan_metrics.history['loss'])))
    
    #text_file = open(log_dir+"\\training_log.txt", "a")
    #text_file.write("Step: %d Disc: real loss: %f fake loss: %f GAN loss: %f\n" % (tot_step, dis_metrics_real[0], dis_metrics_fake[0],gan_metrics[0]))
    #text_file.close()
    avg_GAN_loss.append(np.mean(gan_metrics.history['loss']))
     
    if ((tot_step+1) % 250) == 0:
        print("-----------------------------------------------------------------")
        print("Average Disc_fake loss: %f" % (np.mean(avg_disc_fake_loss)))    
        print("Average Disc_real loss: %f" % (np.mean(avg_disc_real_loss)))    
        print("Average GAN loss: %f" % (np.mean(avg_GAN_loss)))
        print("-----------------------------------------------------------------")
        discriminator.trainable = True
        generator.trainable = True
        generator.save(save_model_dir+str(tot_step)+"_GENERATOR_weights_and_arch.hdf5")
        discriminator.save(save_model_dir+str(tot_step)+"_DISCRIMINATOR_weights_and_arch.hdf5")
OBSERB.close()


#generator = load_model(save_model_dir+'9999_GENERATOR_weights_and_arch.hdf5')

#generate final sample images
# for i in range(10):
#     generate_images(generator, img_save_dir)


"""
#Display Training images sample
save_img_batch(sample_from_dataset(batch_size, image_shape, data_dir = data_dir),img_save_dir+"_12TRAINimage.png")
"""

#Generating GIF from PNG
# images = []
# all_data_dirlist = list(glob.glob(img_save_dir+"*_image.png"))
# for filename in all_data_dirlist:
#     img_num = filename.split('\\')[-1][0:-10]
#     if (int(img_num) % 100) == 0:
#         images.append(imageio.imread(filename))
# imageio.mimsave(img_save_dir+'movie.gif', images) 
    
"""
Alternate way to convert PNG to GIF (ImageMagick):
    >convert -delay 10 -loop 0 *_image.png animated.gif
"""


# In[ ]:



