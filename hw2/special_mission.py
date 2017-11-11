import numpy as np
import pandas as pd
import pickle
import json
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.core import Dropout, Dense, Activation, Flatten, RepeatVector
from keras.layers import Conv1D, MaxPooling1D, Masking
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
import sys
import tensorflow as tf
np.random.seed(42)
tf.set_random_seed(1234)

import os
os.environ['PYTHONHASHSEED'] = '0'

import random as rn
rn.seed(12345)

from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


DIRECTORY = sys.argv[1]
OUTPUT = sys.argv[2]
 
def load_Test_Dataframe(df, label_json, feature_direc):
    Id = []
    Captions = []
    Features = []
    for i in label_json:
        Id.append(i["id"])
        Features.append(np.load(feature_direc + i["id"] + ".npy"))
    df["Id"] = Id
    df["Features"] = Features

def decode_sequence(input_seq,model):
 
    target_seq = np.zeros((len(input_seq), 80))
    target_seq[:, :] = tokenizer.word_index['bos']

    stop_condition = False
    decoded_sentence = [[] for i in range(len(input_seq))]
    
    timestep = 1
    while not stop_condition:
        output_tokens = model.predict(
            [input_seq,target_seq], batch_size=128)

        for i in range(len(input_seq)):
            sampled_token_index = np.argmax(output_tokens[i, timestep-1, :])
            target_seq[i, timestep:] = sampled_token_index
            decoded_sentence[i].append(int(sampled_token_index))
             
        max_decoder_seq_length = 42
        if ( len(decoded_sentence[0]) > max_decoder_seq_length):
            stop_condition = True

        timestep += 1
    return decoded_sentence

Test_Data = pd.DataFrame(columns=['Id', 'Features'])
json_data = open(os.path.join(DIRECTORY, 'testing_label.json')).read()
test_labels = json.loads(json_data)
load_Test_Dataframe(Test_Data, test_labels,os.path.join(DIRECTORY, 'testing_data\\feat\\'))


from keras.models import model_from_json
json_file = open('model_special.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model_special.h5')


with open('tokenizer_special.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

X_test = np.asarray(list(Test_Data['Features']))
X_test = np.asarray(X_test)
X_test_id = list(Test_Data['Id'])

seq_index = 0
decoded_sentence = []
decoded_sentence.append(decode_sequence(X_test,loaded_model))


out = open(OUTPUT,"w")
id_list = ["klteYv1Uv9A_27_33.avi", 
           "5YJaS2Eswg0_22_26.avi", 
           "UbmZAe5u5FI_132_141.avi", 
           "JntMAcTlOF0_50_70.avi", 
           "tJHUH9tpqPg_113_118.avi"]

for i,c in zip(decoded_sentence[0],X_test_id):
    if c in id_list:
        out.write(c+",")
        s = ""
        existed = {'bos':1,'eos':1}
        for idx in i:
            for k,v in tokenizer.word_index.items():
                if idx == v:
                    if k not in existed:
                        s += k + " "
        out.write(s+"\n")
        print(s)
out.close()
