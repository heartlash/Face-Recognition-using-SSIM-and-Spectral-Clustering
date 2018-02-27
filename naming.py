import os
import datetime
import cv2
import numpy as np
import ssim.ssimlib as pyssim
from skimage.measure import compare_ssim as ssim
from sklearn.cluster import SpectralClustering, AffinityPropagation
from sklearn import metrics

face_dir = '/root/Desktop/cprmi/orl_full'
count=0
for file in os.listdir(face_dir):
	print "filename : ",file,"\n"
	
	for i in os.listdir(face_dir+'/'+ file):
		print "before if : ",i
		print i.split('.')[0]
		
		if int(i.split('.')[0])<=5:
			print i
		
			os.system("cp "+face_dir+'/'+file+"/"+i+" /root/Desktop/cprmi/5each_full/"+file+"_"+i)
		
	count=count+1
	
	if count==0:
		exit(0)
