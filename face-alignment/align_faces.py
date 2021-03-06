# USAGE
# python align_faces.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import os


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)
print "shit"
imagelist=os.listdir("/root/Desktop/cprmi/faces")
print len(imagelist)
# load the input image, resize it, and convert it to grayscale

for i in imagelist:
	#print "printing i here : ",i,"\n"
	#print type(i)
	image = cv2.imread("/root/Desktop/cprmi/faces/"+i)
	image = imutils.resize(image, width=800)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# show the original input image and detect faces in the grayscale
# image
#cv2.imshow("Input", image)
	rects = detector(gray, 2)

# loop over the face detections
	for rect in rects:
		# extract the ROI of the *original* face, then align the face
		# using facial landmarks
		(x, y, w, h) = rect_to_bb(rect)
		#print "printing w here : ",w,"\n"
		try:
			w=w+400
			faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
			faceAligned = fa.align(image, gray, rect)

			cv2.imwrite("/root/Desktop/cprmi/faces_align/"+i, faceAligned)

		except:
			pass
