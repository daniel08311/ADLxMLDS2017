import numpy as np

def simple_bbox(heatmap,thres):
	heatarr = np.copy(heatmap)
	points = np.where(heatmap > thres)
	# heatarr[points] = 0.0

	bboxs = [points[0].min(),points[1].min(),points[0].max(),points[1].max()]
	bboxs = [bboxs[1]*4.0,bboxs[0]*4.0,(bboxs[3]-bboxs[1])*4.0,(bboxs[2]-bboxs[0])*4.0]
	return heatmap,bboxs

def smooth(heatmap,k):
	if k > 1:
		return smooth(smooth(heatmap, k-1), 1 )
	elif k == 1:
		copymap = np.zeros_like(heatmap)

		for i in range(0,copymap.shape[0]):
			for j in range(0,copymap.shape[1]):
				adj_matrix = [[i,j],[max(i-1,0),j],[min(i+1,copymap.shape[0]-1),j],[i,max(j-1,0)],[i,min(j+1,copymap.shape[1]-1)]]
				adj_sum = 0.0
				for ix in adj_matrix:
					adj_sum += heatmap[ix[0],ix[1]]
				copymap[i,j] = adj_sum/5.0
		
		copymap = copymap/copymap.std()
		copymap = copymap/copymap.max()
		return copymap
	else :
		return heatmap

def smooth_bbox(heatmap,k,thres):
	heatarr = smooth(heatmap,k)

	return simple_bbox(heatarr,thres)

if __name__ == '__main__':
	pass