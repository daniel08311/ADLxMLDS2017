from keras.models import load_model,model_from_json
import json
import numpy as np
import os

from test_utils import load_img,heatmap2bbox

imgs = open("id_in.txt","r").read().split("\n")
imgs = [p for p in imgs if p != '']

f = open("result.txt","w")

classifier = model_from_json(open("boss_model_resnet_multi.json",'r').read())
# classifier.summary()
classifier.load_weights("boss_model_resnet_multi.h5")

smooth_k = 5
thres = 0.95
size = (256,256)

class_dict = dict(zip(range(0,8),["Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule","Pneumonia","Pneumothorax"]) )

for img in imgs:
    print(img)
    img_data = load_img(img,size,mode="RGB")
    probs = classifier.predict(img_data)

    pred_classes = list(np.where(probs[0] > 0.5)[0])
    bbox_count = len(pred_classes)
    bboxs = []
    output = "%s %d\n" % (img, bbox_count)
    f.write(output)

    for c in pred_classes:
        heatmap ,bbox = heatmap2bbox(classifier,img_data,np.array([c]),thres,smooth_k)
        bboxs.append([c,bbox])

    for box in bboxs:
        output = "%s %f %f %f %f\n" % (class_dict[box[0]], box[1][0], box[1][1], box[1][2], box[1][3])
        f.write(output)

os.remove("id_in.txt")