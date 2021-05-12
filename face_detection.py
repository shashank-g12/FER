import numpy as np
import cv2
import dlib

def face_detect(x_train):
	xnew_train = []
	dnnFaceDetector = dlib.cnn_face_detection_model_v1("/home/shashank/my_project/mmod_human_face_detector.dat")

	rectss = dnnFaceDetector(x_train, 1)
	print(len(rectss))
	for (i,rects) in enumerate(rectss):
		if len(rects) != 0:
			for (j, rect) in enumerate(rects):
				x1 = rect.rect.left()
				y1 = rect.rect.top()
				x2 = rect.rect.right()
				y2 = rect.rect.bottom()
				x1 = 0 if x1 < 0 else x1  
				y1 = 0 if y1 < 0 else y1 
				roi = x_train[i][y1 : y2 + 1, x1 : x2 + 1]
				resized = cv2.resize(roi, (48,48), interpolation = cv2.INTER_AREA)
				xnew_train.append(resized)
		else:
			xnew_train.append(x_train[i])
	return xnew_train