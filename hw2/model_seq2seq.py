import numpy as np
import pandas as pd
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
import itertools
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import seq2seq
from seq2seq.models import AttentionSeq2Seq
from keras.models import Model
from keras.layers import Input
from keras.utils.vis_utils import plot_model

np.random.seed(123)

import tensorflow as tf
tf.set_random_seed(123)

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import random as rn

import os
os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(42)

rn.seed(12345)


session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

def load_Dataframe(df, label_json, feature_direc):
    Id = []
    Captions = []
    Features = []
    for i in label_json:
        Id.append(i["id"])
        Features.append(np.load(feature_direc + i["id"] + ".npy"))
        captions = []
        for k, caption in enumerate(i["caption"]):
            captions.append("bos " + caption + " eos")
        Captions.append(captions)
    df["Id"] = Id
    df["Caption"] = Captions
    df["Features"] = Features
    
def load_Test_Dataframe(df, label_json, feature_direc):
    Id = []
    Captions = []
    Features = []
    for i in label_json:
        Id.append(i["id"])
        Features.append(np.load(feature_direc + i["id"] + ".npy"))
    df["Id"] = Id
    df["Features"] = Features

Train_Data = pd.DataFrame(columns=["Id", "Caption", "Features"])
Test_Data = pd.DataFrame(columns=["Id", "Features"])


json_data = open('training_label.json').read()
train_labels = json.loads(json_data)
json_data = open('testing_label.json').read()
test_labels = json.loads(json_data)


load_Dataframe(Train_Data, train_labels, "training_data\\feat\\")
load_Test_Dataframe(Test_Data, test_labels,"testing_data\\feat\\")


filters = "(,\n].;)”’“&'" + '"' + "'"
tokenizer = Tokenizer(num_words=1000,filters=filters)
All = [j for i in list(Train_Data["Caption"]) for j in i ]
tokenizer.fit_on_texts(All)


from keras.layers import Permute
from keras.layers import merge, Concatenate, Merge, add, Masking, Activation, dot, multiply, concatenate
from keras import backend as K
from keras.optimizers import Adam, RMSprop

latent_dim = 1024
num_decoder_tokens = 1001
batch_size = 12
epochs = 2

video_model_input = Input(shape=(80, 4096))
video_model, h1,h2 = LSTM(4096, return_sequences=True, return_state=True, activation="tanh")(video_model_input)

caption_model_input = Input(shape=(None,))
caption_embed = Embedding(num_decoder_tokens, 300)(caption_model_input)
decoder = LSTM(4096,return_sequences=True, stateful=False, activation="tanh")(caption_embed, initial_state=[h1,h2])

score = dot([decoder,video_model],2)
score = Activation('softmax')(score)
score = Dropout(0.35)(score)
permute = Permute((2,1))(video_model_input)
attention = dot([score,permute],2)

final = concatenate([attention, decoder])
final = Activation('tanh')(final)
final = Dropout(0.35)(final)
final = Dense(num_decoder_tokens, activation='softmax')(final)

model = Model([video_model_input,caption_model_input],final)
model.compile(optimizer=Adam(lr=0.0001, clipnorm=1.), loss='categorical_crossentropy')
print(model.summary())


for n in range(8):
    All_one_hot = []
    no_one_hot = []
    for sample in list(Train_Data["Caption"]):
        r = n % len(sample)
        no_one_hot.append(pad_sequences(tokenizer.texts_to_sequences([sample[r]]), maxlen = 42, padding='post'))
        All_one_hot.append(to_categorical(pad_sequences(tokenizer.texts_to_sequences([sample[r]]), maxlen = 42, padding='post'),num_classes = num_decoder_tokens))

    Y_train = np.asarray(All_one_hot)
    X_train = np.asarray(list(Train_Data["Features"]))

    encoder_input_data = X_train
    decoder_input_data = np.asarray(no_one_hot).reshape((1450,42))[:,0:-1]
    decoder_target_data = Y_train[:,1:,:]

    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2, verbose=1)


def beam_search_predictions(model, image, beam_index = 3):
    max_len=42
    start = [tokenizer.word_index["bos"]]
    
    start_word = [[start, 0.0]]
     
    target_seq = np.zeros((1, 1, 501))
    target_seq[0, 0, tokenizer.word_index['bos']] = 1.
    targets = []
    for i in range(beam_index):
        targets.append(target_seq)
    
    timestep = 0
    first = True
    while len(start_word[0][0]) < max_len:
        temp = []
        for idx, s in enumerate(start_word):
            # Populate the first character of target sequence with the start character.
            target_seq = targets[idx]
            target_seq[0, timestep, s[0]] = 1.
            preds = model.predict([input_seq, target_seq])
            word_preds = np.argsort(preds[0][-1])[-beam_index:]
            # Getting the top <beam_index>(n) predictions and creating a 
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][-1][w]
                temp.append([next_cap, prob])
            
            if not first:
                targets[idx] = np.zeros((1,timestep+2,501))
                targets[idx][0,:timestep+1,:] = target_seq
        
        #print(targets[0][0].shape)
        
        if first:
            for i in range(len(targets)):
                targets[i] = np.zeros((1,timestep+2,501))
                targets[i][0,:timestep+1,:] = target_seq
        
        timestep+=1
        
        first = False   
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]

    start_word = start_word[-1][0]
    decode_seq = []
    for i in start_word:
        if tokenizer.word_index['eos'] != i :
            decode_seq.append(i)
        else:
            break

    return [decode_seq]


def decode_sequence(input_seq):
 
    target_seq = np.zeros((len(input_seq), 1))
    target_seq[:, 0] = tokenizer.word_index['bos']
    last_seq = target_seq
    stop_condition = False
    decoded_sentence = [[] for i in range(len(input_seq))]
    
    timestep = 1
    while not stop_condition:
        output_tokens = model.predict(
            [input_seq,target_seq], batch_size=10)
        
        target_seq = np.zeros((len(input_seq), timestep+1))
        target_seq[:,:timestep] = last_seq
        for i in range(len(input_seq)):
            sampled_token_index = np.argmax(output_tokens[i, timestep-1, :])
            target_seq[i, timestep] = sampled_token_index
            decoded_sentence[i].append(int(sampled_token_index))
        
        max_decoder_seq_length = 42
        if (  len(decoded_sentence[0]) > max_decoder_seq_length):
            stop_condition = True
             
        last_seq = target_seq
        timestep += 1

    return decoded_sentence


X_test = np.asarray(list(Test_Data["Features"]))
X_test = np.asarray(X_test)
X_test_id = list(Test_Data["Id"])


seq_index = 0
decoded_sentence = []
input_seq = X_test
decoded_sentence.append(decode_sequence(input_seq))


out = open("output.txt","w")
for i,c in zip(decoded_sentence[0],X_test_id):
    out.write(c+",")
    s = ""
    existed = {'bos':1,'eos':1}
    for idx in i:
        for k,v in tokenizer.word_index.items():
            if idx == v:
                if k not in existed:
                    s += k + " "
                    #existed[k] = 1
    out.write(s[:s.rfind("eos")]+"\n")
    print(s[:s.rfind("eos")])
out.close()


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")


# from keras.models import model_from_json
# json_file = open('model_special.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model_special.h5")
# print("Loaded model from disk")



import pickle
with open('model.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)



