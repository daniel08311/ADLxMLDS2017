file_48_39 = open("48_39.map")
file_39_phone = open("48phone_char.map")

map_48_39 = {}
for line in file_48_39:
    mapping = line.split("\t")
    map_48_39[mapping[0]] = mapping[1][:-1] 


map_39_phone = {}
for line in file_39_phone:
    mapping = line.split("\t")
    map_39_phone[mapping[0]] = mapping[2][:-1] 

y_tokens = []
for k,v in map_48_39.items():
    y_tokens.append(map_39_phone[v])
y_tokens = list(sorted(set(y_tokens))) 


map_phone_index = {}
for i, token in zip(range(len(y_tokens)),y_tokens):
    map_phone_index[token] = i+1


import pandas as pd
import numpy as np
X_file = open("mfcc/train.ark")
X_test_file = open("mfcc/test.ark")
Y_file = open("train.lab")

X = pd.DataFrame(columns=["Id","Feature"])
Ids = []
Features = []
for line in X_file:
    split = line.split(" ")
    features = []
    for feats in split[1:]:
        features.append(feats)
    features[-1] = features[-1][:-1]
    Features.append(np.asarray(features,dtype=float))
    Ids.append(split[0])
X["Id"] = Ids
X["Feature"] = Features

Y = pd.DataFrame(columns=["Id","Phone"])
Ids = []
Phones = []
for line in Y_file:
    split = line.split(",")
    Ids.append(split[0])
    Phones.append(split[1][:-1])
Y["Id"] = Ids
Y["Phone"] = [map_39_phone[map_48_39[i]] for i in Phones] 

Concat_XY = pd.merge(X, Y, on='Id')
raw_features = list(Concat_XY["Feature"])
raw_phones = list(Concat_XY["Phone"])
raw_Ids = list(Concat_XY["Id"])

Concat_frames = pd.DataFrame(columns=["Id","Sequence","Feature"])
Exist_id = {}
Ids = []
for i in list(Concat_XY["Id"]):
    Id = i[:i.rfind("_")] 
    if Id not in Exist_id:
        Ids.append(Id)
        Exist_id[Id] = 1
Concat_frames["Id"] = Ids
Concat_frames["Sequence"] = ""
Concat_frames["Feature"] = ""

Concat_frames = Concat_frames.set_index("Id")


for i, feat, phone in zip(raw_Ids,raw_features,raw_phones) :
    Id = i[:i.rfind("_")]
    row = Concat_frames.loc[Id] 
    if row["Sequence"] == "":
        row["Sequence"] = phone
        row["Feature"] = feat
        
    else :
        row["Sequence"] = str(row["Sequence"]+phone)
        row["Feature"] = np.column_stack((row["Feature"],feat))

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Activation, GRU, TimeDistributed, Conv1D, MaxPooling1D, MaxPooling2D, Flatten, Dropout, SimpleRNN, Bidirectional, Convolution2D, Permute, BatchNormalization, RepeatVector, Input, MaxPool2D 
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers
from keras import backend as K
from keras import initializers
from keras import utils 
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
import pickle
K.set_image_dim_ordering("th") 
K.set_image_data_format("channels_last")

scaler = StandardScaler()

X_train = list(Concat_frames['Feature'])
MAX_SEQ = np.max(np.asarray([len(i[0]) for i in  X_train]))
for i in range(len(X_train)):
    shape = X_train[i].shape[1]
    X_train[i] = scaler.fit_transform(X_train[i].transpose())
    zeros = np.zeros((MAX_SEQ-shape,X_train[i].shape[1]))
    X_train[i] = np.row_stack((X_train[i],zeros))

X_train = np.asarray(X_train,dtype=float)

Y_train = list(Concat_frames["Sequence"])
for i in range(len(Y_train)):
    Y_train[i] = [map_phone_index[j] for j in Y_train[i]]
    Y_train[i] = utils.to_categorical(Y_train[i],num_classes=40)
    shape = Y_train[i].shape[0]
    zeros = np.zeros((MAX_SEQ-shape,40))
    Y_train[i] = np.row_stack((Y_train[i],zeros))
Y_train = np.asarray(Y_train,dtype=float)    


X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2]))
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.15, random_state=1)

model = Sequential()
model.add(Bidirectional(GRU(512, return_sequences=True, dropout=0.5, recurrent_dropout=0.5,activation='tanh'),input_shape=(X_train.shape[1],X_train.shape[2])))
model.add(Bidirectional(GRU(256, return_sequences=True, dropout=0.5, recurrent_dropout=0.5,activation='tanh')))
model.add(Bidirectional(GRU(128, return_sequences=True, dropout=0.5, recurrent_dropout=0.5,activation='tanh')))
model.add(Bidirectional(GRU(128, return_sequences=True, dropout=0.5, recurrent_dropout=0.5,activation='tanh')))

model.add(TimeDistributed(Dense(40,activation='softmax')))
opt = optimizers.adam(lr=0.0005)

model.compile(loss='categorical_crossentropy', optimizer=opt)
print(model.summary())

checkpoint = ModelCheckpoint("test.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
earlystop = EarlyStopping(monitor='val_loss', patience=40, verbose=1, mode="min")
callbacks_list = [checkpoint,earlystop]

model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=1, callbacks=callbacks_list, batch_size=70, verbose=1)
model_json = model.to_json()
with open("test.json", "w") as json_file:
    json_file.write(model_json)

X_test_df = pd.DataFrame(columns=["Id","Feature"])
Ids = []
Features = []
for line in X_test_file:
    split = line.split(" ")
    features = []
    for feats in split[1:]:
        features.append(feats)
    features[-1] = features[-1][:-1]
    Features.append(np.asarray(features,dtype=float))
    Ids.append(split[0])
X_test_df["Id"] = Ids
X_test_df["Feature"] = Features

X_test_df_concat = pd.DataFrame(columns=["Id","Feature"])
Exist_id = {}
Ids = []
for i in list(X_test_df["Id"]):
    Id = i[:i.rfind("_")] 
    if Id not in Exist_id:
        Ids.append(Id)
        Exist_id[Id] = 1
X_test_df_concat["Id"] = Ids
X_test_df_concat["Feature"] = ""

X_test_df_concat = X_test_df_concat.set_index("Id")

for i, feat in zip(list(X_test_df["Id"]),list(X_test_df["Feature"])) :
    Id = i[:i.rfind("_")]
    row = X_test_df_concat.loc[Id]
    if row["Feature"] == "":
        row["Feature"] = feat
    else :
        row["Feature"] = np.column_stack((row["Feature"],feat))

X_test = list(X_test_df_concat['Feature'])
for i in range(len(X_test)):
    shape = X_test[i].shape[1]
    zeros = np.zeros((MAX_SEQ-shape,39))
    X_test[i] = scaler.fit_transform(X_test[i].transpose())
    X_test[i] = np.row_stack((X_test[i],zeros))
    
X_test = np.asarray(X_test,dtype=float)

result = model.predict_classes(X_test)

output = []
for i in result:
    string = "0"
    for j in i:
        for k,v in map_phone_index.items():
            if j == v:
                #if string[-1] != k:
                string += k
    string = string[1:]
    output.append(string)

testing = []
for i in output:
    test = "0"
    for idx in range(len(i)):
        if idx != 0 and idx != len(i)-1:
            if (i[idx] != i[idx-1]) and (i[idx] == i[idx+1]):
                test += i[idx]
    testing.append(test[1:-1])

X_test_df_concat = X_test_df_concat.reset_index(level=['Id'])

output_file = open("test.csv",'w')
output_file.write("id,phone_sequence\n")
for i, j in zip(list(X_test_df_concat["Id"]),testing):
    output_file.write(i+","+j+"\n")
output_file.close()

