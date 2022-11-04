import utiles
import argparse
import imutils
import time
import dlib
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image")
ap.add_argument("-ef", "--end_folder", type=str, required=True,
	help="path to output (croped) images")
ap.add_argument("-m", "--model", type=str,
	default="mmod_human_face_detector.dat",
	help="path to dlib's CNN face detector model")
ap.add_argument("-u", "--upsample", type=int, default=1,
	help="# of times to upsample")
args = vars(ap.parse_args())

# Loading the Face detector with the pre-trained model
detector = dlib.cnn_face_detection_model_v1(args["model"])

img = cv2.imread(args["image"])
img = imutils.resize(img, width=600)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detection
start = time.time()
print('Face detection with CNN started')
rects = detector(rgb, args["upsample"]) # this is a list
end = time.time()
print(f'Run time {end-start}')

# Fix the boxes
boxes = [utiles.trim_n_convert_v2(image=img, epsilon = 5, rectangle=r.rect) for r in rects]

# To store images in the desired folder
os.chdir(args["end_folder"])
box_count = 0
# loop over the bounding boxes
for (x, y, w, h) in boxes:
	# draw the bounding box on our image
	fcs = img[y:y+h, x:x+w]
	fcs = imutils.resize(fcs, width=100)
	cv2.imwrite('face'+str(box_count)+'.jpg', fcs)
	cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
	box_count += 1

# show the output image
cv2.imshow("Output", img)
cv2.waitKey(5000)

# Going back to root directory
os.chdir('/Users/pauor506/Documents/CDHU/June2022:Dec2022/Wasp/Wasp_DataCollection') # TO BE CHANGED ACCORDINGLY