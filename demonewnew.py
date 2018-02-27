import os
import numpy as np
import cv2
import main as tv
#from matplotlib import pyplot as plt

DIR_NAME = 'faces_blurred'

true_lab_nums=len(os.listdir(DIR_NAME))

# Demo for clustering a set of 20 images using 'imgcluster' module.
# To be executed in the standalone mode by default. IP[y] Notebook requires some minor adjustments.

""" True (reference) labels for the provided images - defined manually according to the semantic
    meaning of images. For example: bear, raccoon and fox should belong to the same cluster.
    Please feel free to change the true labels according to your perception of these images  :-)
"""
TRUE_LABELS = [1]*true_lab_nums

al=['SIFT']

if __name__ == "__main__":
    	
	for each in al:

		c= tv.do_cluster(DIR_NAME, algorithm=each,print_metrics=True, labels_true=TRUE_LABELS )
		num_clusters = len(set(c))
		images = os.listdir(DIR_NAME)
	
		clusters = []
		for n in range(num_clusters):
		#print("\n --- Images from cluster #%d ---" % n)
			imgs = []
			for i in np.argwhere(c == n):
				if i != -1:
					#print(images[int(i)])
					inp = images[int(i)].split("_")
					inp2 = inp[0]
					imgs.append(int(inp2))

			clusters.append(imgs)

		tp = 0
		tn = 0
		fp = 0
		fn = 0
		cluster2 = []
		for i in clusters:
			print("cluster = ",i)
			d = []
			for j in range(1,41):
				count = i.count(j)
				d.append([count,j])
			d.sort()
			d.reverse()
			print("d = ",d)
			cluster2.append(d)
			tp += d[0][0]
			for k in range(1,len(d)):
				fp += d[k][0]

		for i in range(1,41):
			for j in cluster2:
				if j[0][1]!=i:
					for k in j:
						if k[1]!=i:
							tn += k[0]
						else:
							fn += k[0]
			
		print("tp = ",tp)
		print("tn = ",tn)
		print("fp = ",fp)
		print("fn = ",fn)

	
		acc = (tp+tn)/(tp+tn+fp+fn)
		pre = tp/(tp+fp)
		rec = tp/(tp+fn)

		print("ACCURACY = ")
		print(acc)
		print("PRECISION = ")
		print(pre)
		print("RECALL = ")
		print(rec)
	
		
        

