import pandas as pd
from scipy import misc
import cv2
import numpy as np
import pickle
data = pd.read_csv("train_multi.csv")
data_test = pd.read_csv("valid_multi.csv")
data = pd.get_dummies(data,columns=['Gener','Position'])
data_test = pd.get_dummies(data_test,columns=['Gener','Position'])

Y = data[['Atelectasis', 'Cardiomegaly',
          'Effusion', 'Infiltration', 
          'Mass', 'Nodule', 'Pneumonia',
          'Pneumothorax']].values
X_meta = data[['Age', 'Gener_F', 'Gener_M', 'Position_AP', 'Position_PA']].values

Y_test = data_test[['Atelectasis', 'Cardiomegaly',
          'Effusion', 'Infiltration', 
          'Mass', 'Nodule', 'Pneumonia',
          'Pneumothorax']].values
X_meta_test = data_test[['Age', 'Gener_F', 'Gener_M', 'Position_AP', 'Position_PA']].values

from keras import applications
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Input, concatenate, merge, Merge
from keras.optimizers import Adam
from keras.models import Model
from keras.models import model_from_json, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.constraints import maxnorm
from sklearn.cross_validation import StratifiedKFold
from keras.layers import Dropout


base_model = applications.resnet50.ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=(256,256,3), pooling='None', classes=8)

base_output = base_model.output
output = Flatten()(base_output)
output = Dropout(0.3)(output)
output = Dense(8, activation='softmax')(output)
model = Model(inputs=[base_model.input], outputs=output)

opt = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

x_test = np.load('valid_multi.npy').astype(float)
x_test = np.stack((x_test,)*3,axis=-1)
x_test/=255.0
for k in range(1):
    for i in range(6):
        x_img = np.load('train_multi_'+str(i)+'.npy').astype(float)
        x_img = np.stack((x_img,)*3,axis=-1)
        x_img /= 255.0
        x_meta = X_meta[i*7500:(i+1)*7500]
        y = Y[i*7500:(i+1)*7500]
        model.fit(x_img,y, batch_size=32, epochs=1, verbose=1, validation_data=(x_test,Y_test))

model_json = model.to_json()
with open("test_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("test_model.h5")
