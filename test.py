import cv2
import sys
import numpy as np
import datetime
import os
import glob
import time
from imutils.video import VideoStream
from retinaface_cov import RetinaFaceCoV

vs = VideoStream(src=0).start()
time.sleep(2.0)

gpuid = 0
detector = RetinaFaceCoV('./model/mnet_cov2', 0, gpuid, 'net3l')
while True:
	scales = [640, 1080]
	thresh = 0.8
	mask_thresh = 0.2
	img = vs.read()
	img = cv2.flip(img, 1)
	im_shape = img.shape
	target_size = scales[0]
	max_size = scales[1]
	im_size_min = np.min(im_shape[0:2])
	im_size_max = np.max(im_shape[0:2])
	im_scale = float(target_size) / float(im_size_min)
	if np.round(im_scale * im_size_max) > max_size:
	    im_scale = float(max_size) / float(im_size_max)
		
	scales = [im_scale]
	flip = False

	faces, landmarks = detector.detect(img,thresh,scales=scales,do_flip=flip)
    	
	if faces is not None:
		print('find', faces.shape[0], 'faces')
		for i in range(faces.shape[0]):
			face = faces[i]
			box = face[0:4].astype(np.int)
			mask = face[5]
			if mask >= mask_thresh:
				color = (0, 255, 0)
			else:
				color = (0, 0, 255)

			cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
			landmark5 = landmarks[i].astype(np.int)

			for l in range(landmark5.shape[0]):
					color = (255, 0, 0)
					cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)
		    
	cv2.imshow("Frame", img)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
        	break
cv2.destroyAllWindows()
vs.stop()

    
