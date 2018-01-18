import os
import cv2
import numpy as np

import scipy.misc

from vis.visualization import visualize_saliency
from bbox_predict import simple_bbox,smooth_bbox

def load_img(img_path,size,mode = "RGB"):
	img = scipy.misc.imread(img_path , mode = "RGB" )
	if mode == "GRAY":
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = scipy.misc.imresize(img, size)
	img = img/255.0
	if mode == "GRAY":
		img = np.expand_dims(img,axis=-1)
	img = np.expand_dims(img,axis=0)
	return img

def heatmap2bbox(classifier,img,pred,thres_box,k):
	layer_idx = [idx for idx, layer in enumerate(classifier.layers)if layer.name == "dense_1"][0]
	
	heatmap = visualize_saliency(classifier, layer_idx, pred, img)
	heatmap = heatmap/heatmap.std()
	heatmap = heatmap/heatmap.max()
	
	heatarr, bbox = smooth_bbox(heatmap,k,thres_box)
	return heatarr,bbox

if __name__ == '__main__':
	pass
