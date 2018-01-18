import judger_medical as judger

from keras.models import load_model,model_from_json
import json
import os
import numpy as np

from test_utils import load_img,heatmap2bbox

imgs = judger.get_file_names()

f = judger.get_output_file_object()

path_batch = []
fout = open("id_in.txt","w")
for p in imgs:
    path_batch.append(p)
    fout.write("%s\n" % p)

    if len(path_batch) >= 10:
        path_batch = []
        fout.close()
        os.system("python3 single_test.py")

        while True:
            if not os.path.isfile("id_in.txt"):
                break

        fout = open("id_in.txt","w")
        fin = open("result.txt","r")
        result = fin.read()
        result = result.encode()
        f.write(result)
        fin.close()

if len(path_batch) > 0:
    path_batch = []
    fout.close()
    os.system("python3 single_test.py")

    while True:
        if not os.path.isfile("id_in.txt"):
            break

    fout = open("id_in.txt","w")
    fin = open("result.txt","r")
    result = fin.read()
    result = result.encode()
    f.write(result)
    fin.close()

if os.path.isfile("id_in.txt"):
    os.remove("id_in.txt")
if os.path.isfile("result.txt"):
    os.remove("result.txt")

score, err = judger.judge()
if err is not None:  # in case we failed to judge your submission
    print(err)