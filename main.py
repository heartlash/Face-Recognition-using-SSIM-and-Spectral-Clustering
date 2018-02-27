# Copyright (c) 2016, Oleg Puzanov
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import os
import datetime
import cv2
import numpy as np
import ssim.ssimlib as pyssim
from skimage.measure import compare_ssim as ssim
from sklearn.cluster import SpectralClustering, AffinityPropagation
from sklearn import metrics,decomposition

# Constant definitions
SIM_IMAGE_SIZE = (640, 480)
SIFT_RATIO = 0.7
AKAZE_RATIO = 0.9
MSE_NUMERATOR = 1000.0
IMAGES_PER_CLUSTER = 5

""" Returns the normalized similarity value (from 0.0 to 1.0) for the provided pair of images.
    The following algorithms are supported:
    * SIFT: Scale-invariant Feature Transform
    * SSIM: Structural Similarity Index
    * CW-SSIM: Complex Wavelet Structural Similarity Index
    * MSE: Mean Squared Error
"""
def get_image_similarity(img1, img2, algorithm='SIFT'):
    	# Converting to grayscale and resizing
	i1 = cv2.resize(cv2.imread(img1, cv2.IMREAD_GRAYSCALE), SIM_IMAGE_SIZE)
	i2 = cv2.resize(cv2.imread(img2, cv2.IMREAD_GRAYSCALE), SIM_IMAGE_SIZE)

	similarity = 0.0

	if algorithm =='SIFT':
		#print("just some stuff")
        	# Using OpenCV for feature detection and matching
		sift = cv2.xfeatures2d.SIFT_create()
		k1, d1 = sift.detectAndCompute(i1, None)
		#print("printing k1 and d1 here : \n",k1,"\n",d1,"\n")
		
		k2, d2 = sift.detectAndCompute(i2, None)
		#print("printing k2 and d2 here : \n",k2,"\n",d2,"\n")
		

		bf = cv2.BFMatcher()      #brute-force matcher
		matches = bf.knnMatch(d1, d2, k=2)
		#print("matches : ",matches,"\n")
		#print("len of matches : ",len(matches))

		for m, n in matches:
			#print(m,n)
			#print("printing now the distance part : \n",m.distance,n.distance,"\n")
			
			if m.distance < SIFT_RATIO * n.distance:
				similarity += 1.0

       		# Custom normalization for better variance in the similarity matrix
		if similarity == len(matches):
			similarity = 1.0
		elif similarity > 1.0:
			similarity = 1.0 - 1.0/similarity
		elif similarity == 1.0:
			similarity = 0.1
		else:
			similarity = 0.0
	elif algorithm == 'CW-SSIM':
       	 	# FOR EXPERIMENTS ONLY!
        	# Very slow algorithm - up to 50x times slower than SIFT or SSIM.
        	# Optimization using CUDA or Cython code should be explored in the future.
		similarity = pyssim.SSIM(img1).cw_ssim_value(img2)


	elif algorithm == 'SSIM':
        	# Default SSIM implementation of Scikit-Image
		similarity = ssim(i1, i2)
		#print("similarity by using ssim",similarity)
	elif algorithm=='ORB':
		orb = cv2.ORB_create()
		k1, d1 = orb.detectAndCompute(i1,None)
		k2, d2 = orb.detectAndCompute(i2,None)	
		
		# create BFMatcher object
		bf = cv2.BFMatcher()

		# Match descriptors.
		matches = bf.knnMatch(d1, d2, k=2)

		# Sort them in the order of their distance.
		#matches = sorted(matches, key = lambda x:x.distance)
		#print("printing matches and its length",matches,"\n",len(matches))
		for m, n in matches:
			#print(m,n)
			#print("printing now the distance part : \n",m.distance,n.distance,"\n")
			
			if m.distance < SIFT_RATIO * n.distance:
				similarity += 1.0

       		# Custom normalization for better variance in the similarity matrix
		if similarity == len(matches):
			similarity = 1.0
		elif similarity > 1.0:
			similarity = 1.0 - 1.0/similarity
		elif similarity == 1.0:
			similarity = 0.1
		else:
			similarity = 0.0

	elif algorithm=='SURF':
		surf = cv2.xfeatures2d.SURF_create()
		k1, d1 = surf.detectAndCompute(i1,None)
		k2, d2 = surf.detectAndCompute(i2,None)	
		
		# create BFMatcher object
		bf = cv2.BFMatcher()

		# Match descriptors.
		matches = bf.knnMatch(d1, d2, k=2)

		# Sort them in the order of their distance.
		#matches = sorted(matches, key = lambda x:x.distance)
		#print("printing matches and its length",matches,"\n",len(matches))
		for m, n in matches:
			#print(m,n)
			#print("printing now the distance part : \n",m.distance,n.distance,"\n")
			
			if m.distance < SIFT_RATIO * n.distance:
				similarity += 1.0

       		# Custom normalization for better variance in the similarity matrix
		if similarity == len(matches):
			similarity = 1.0
		elif similarity > 1.0:
			similarity = 1.0 - 1.0/similarity
		elif similarity == 1.0:
			similarity = 0.1
		else:
			similarity = 0.0

	elif algorithm=='BRISK':
		brisk = cv2.BRISK_create()
		k1, d1 = brisk.detectAndCompute(i1,None)
		k2, d2 = brisk.detectAndCompute(i2,None)	
		
		# create BFMatcher object
		bf = cv2.BFMatcher()

		# Match descriptors.
		matches = bf.knnMatch(d1, d2, k=2)

		# Sort them in the order of their distance.
		#matches = sorted(matches, key = lambda x:x.distance)
		#print("printing matches and its length",matches,"\n",len(matches))
		for m, n in matches:
			#print(m,n)
			#print("printing now the distance part : \n",m.distance,n.distance,"\n")
			
			if m.distance < SIFT_RATIO * n.distance:
				similarity += 1.0

       		# Custom normalization for better variance in the similarity matrix
		if similarity == len(matches):
			similarity = 1.0
		elif similarity > 1.0:
			similarity = 1.0 - 1.0/similarity
		elif similarity == 1.0:
			similarity = 0.1
		else:
			similarity = 0.0

	
	elif algorithm=='AKAZE':
		akaze = cv2.AKAZE_create()
		k1, d1 = akaze.detectAndCompute(i1,None)
		k2, d2 = akaze.detectAndCompute(i2,None)	
		
		# create BFMatcher object
		bf = cv2.BFMatcher()

		# Match descriptors.
		matches = bf.knnMatch(d1, d2, k=2)

		# Sort them in the order of their distance.
		#matches = sorted(matches, key = lambda x:x.distance)
		#print("printing matches and its length",matches,"\n",len(matches))
		for m, n in matches:
			#print(m,n)
			#print("printing now the distance part : \n",m.distance,n.distance,"\n")
			
			if m.distance < AKAZE_RATIO * n.distance:
				similarity += 1.0

       		# Custom normalization for better variance in the similarity matrix
		if similarity == len(matches):
			similarity = 1.0
		elif similarity > 1.0:
			similarity = 1.0 - 1.0/similarity
		elif similarity == 1.0:
			similarity = 0.1
		else:
			similarity = 0.0

	elif algorithm=='KAZE':
		kaze = cv2.KAZE_create()
		k1, d1 = kaze.detectAndCompute(i1,None)
		k2, d2 = kaze.detectAndCompute(i2,None)	
		
		# create BFMatcher object
		bf = cv2.BFMatcher()

		# Match descriptors.
		matches = bf.knnMatch(d1, d2, k=2)

		# Sort them in the order of their distance.
		#matches = sorted(matches, key = lambda x:x.distance)
		#print("printing matches and its length",matches,"\n",len(matches))
		for m, n in matches:
			#print(m,n)
			#print("printing now the distance part : \n",m.distance,n.distance,"\n")
			
			if m.distance < AKAZE_RATIO * n.distance:
				similarity += 1.0

       		# Custom normalization for better variance in the similarity matrix
		if similarity == len(matches):
			similarity = 1.0
		elif similarity > 1.0:
			similarity = 1.0 - 1.0/similarity
		elif similarity == 1.0:
			similarity = 0.1
		else:
			similarity = 0.0





	else:
        	# Using MSE algorithm with custom normalization
		err = np.sum((i1.astype("float") - i2.astype("float")) ** 2)
		err /= float(i1.shape[0] * i2.shape[1])

		if err > 0.0:
			similarity = MSE_NUMERATOR / err
		else:
			similarity = 1.0

	return similarity

# Fetches all images from the provided directory and calculates the similarity
# value per image pair.
def build_similarity_matrix(dir_name, algorithm='SIFT'):
	images = os.listdir(dir_name)
	#print("printing images here : ",images,"\n")
	
	num_images = len(images)
	#print("number of images here : ",num_images,"\n")
	sm = np.zeros(shape=(num_images, num_images), dtype=np.float64)
	#print("sm before filling diagonal : ",sm,"\n")
	#print(sm.size)
	np.fill_diagonal(sm, 1.0)
	#print("sm after filling diagonal : ",sm,"\n")

	print("Building the similarity matrix using %s algorithm for %d images" %
          (algorithm, num_images))
	start_total = datetime.datetime.now()

    	# Traversing the upper triangle only - transposed matrix will be used
    	# later for filling the empty cells.
	k = 0
	print("sm.shape[0] here : ",sm.shape[0],"  ",sm.shape[1],"\n")
	for i in range(sm.shape[0]):
		for j in range(sm.shape[1]):
			j = j + k
			if i != j and j < sm.shape[1]:
				sm[i][j] = get_image_similarity('%s/%s' % (dir_name, images[i]),
                                                '%s/%s' % (dir_name, images[j]),
                                                algorithm=algorithm)
				#print("(",i,",",j,")")
				#print(sm[i][j])
		k += 1

    	# Adding the transposed matrix and subtracting the diagonal to obtain
    	# the symmetric similarity matrix
	sm = sm + sm.T - np.diag(sm.diagonal())

	end_total = datetime.datetime.now()
	print("Done - total calculation time: %d seconds" % (end_total - start_total).total_seconds())
	return sm

""" Returns a dictionary with the computed performance metrics of the provided cluster.
    Several functions from sklearn.metrics are used to calculate the following:
    * Silhouette Coefficient
      Values near 1.0 indicate that the sample is far away from the neighboring clusters.
      A value of 0.0 indicates that the sample is on or very close to the decision boundary
      between two neighboring clusters and negative values indicate that those samples might
      have been assigned to the wrong cluster.
    * Completeness Score
      A clustering result satisfies completeness if all the data points that are members of a
      given class are elements of the same cluster. Score between 0.0 and 1.0. 1.0 stands for
      perfectly complete labeling.
    * Homogeneity Score
      A clustering result satisfies homogeneity if all of its clusters contain only data points,
      which are members of a single class. 1.0 stands for perfectly homogeneous labeling.
"""
def get_cluster_metrics(X, labels, labels_true=None):
	metrics_dict = dict()
	metrics_dict['Silhouette coefficient'] = metrics.silhouette_score(X,
                                                                      labels,
                                                                      metric='precomputed')
	if labels_true:
		metrics_dict['Completeness score'] = metrics.completeness_score(labels_true, labels)
		metrics_dict['Homogeneity score'] = metrics.homogeneity_score(labels_true, labels)
	
	#print("printing the metrics dict : \n",metrics_dict,"\n")

	return metrics_dict

""" Executes two algorithms for similarity-based clustering:
    * Spectral Clustering
    * Affinity Propagation
    ... and selects the best results according to the clustering performance metrics.
"""
def do_cluster(dir_name, algorithm='SIFT', print_metrics=True, labels_true=None):
	matrix = build_similarity_matrix(dir_name, algorithm=algorithm)
	#print("printing matrix",matrix,"\n\n")

	sc = SpectralClustering(n_clusters=int(matrix.shape[0]/IMAGES_PER_CLUSTER),
                            affinity='precomputed').fit(matrix)
	#print("printing special cluster matrix",sc,"\n\n")
	#print("printing sc labels : ",sc.labels_)
	sc_metrics = get_cluster_metrics(matrix, sc.labels_, labels_true)

	if print_metrics:
		print("\nPerformance metrics for Spectral Clustering")
		print("Number of clusters: %d" % len(set(sc.labels_)))
		for k in list(sc_metrics.keys()):
			print("%s: %.2f" % (k, sc_metrics[k])) 

	af = AffinityPropagation(affinity='precomputed').fit(matrix)
	af_metrics = get_cluster_metrics(matrix, af.labels_, labels_true)

	if print_metrics:
		print("\nPerformance metrics for Affinity Propagation Clustering")
		print("Number of clusters: %d" % len(set(af.labels_)))
		for k in list(af_metrics.keys()):
			 print("%s: %.2f" % (k, af_metrics[k]))


	if (sc_metrics['Silhouette coefficient'] >= af_metrics['Silhouette coefficient']) and \
            (sc_metrics['Completeness score'] >= af_metrics['Completeness score'] or
             sc_metrics['Homogeneity score'] >= af_metrics['Homogeneity score']):
		print("\nSelected Spectral Clustering for the labeling results")
		print("printing the final sc.labels_",sc.labels_)
		return sc.labels_
	else:
		print("\nSelected Affinity Propagation for the labeling results")
	print("printing the final sc.labels_",sc.labels_)
	print("type = ",type(af.labels_))


		

	return sc.labels_
