import pandas as pd
import numpy as np
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
import sys
K.set_image_dim_ordering("th") 
K.set_image_data_format("channels_last")

data_directory = sys.argv[1]
output_fileName = sys.argv[2]

file_48_39 = open(data_directory+"phones/48_39.map")
file_39_phone = open(data_directory+"48phone_char.map")

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

MAX_SEQ = 777
scaler = StandardScaler()

X_test_file = open(data_directory+"mfcc/test.ark")

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

json_file = open('best_rnn1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights("best_rnn1.hdf5")

json_file = open('best_rnn2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model2 = model_from_json(loaded_model_json)

model2.load_weights("best_rnn2.hdf5")

final = np.zeros((592,777,1))
result = model.predict(X_test)
result2 = model2.predict(X_test)
for i in range(len(result2)):
    for k in range(len(result2[i])):
        final[i][k] = np.argmax((result[i][k] + result2[i][k])/2 )

output = []
for i in final:
    string = "0"
    for j in i:
        for k,v in map_phone_index.items():
            if j == v:
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

output_file = open(output_fileName,'w')
output_file.write("id,phone_sequence\n")
for i, j in zip(list(X_test_df_concat["Id"]),testing):
    output_file.write(i+","+j+"\n")
output_file.close()

