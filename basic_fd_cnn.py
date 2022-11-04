from utiles import trim_n_convert
import argparse
import imutils
import time
import dlib
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image")
ap.add_argument("-m", "--model", type=str,
	default="mmod_human_face_detector.dat",
	help="path to dlib's CNN face detector model")
ap.add_argument("-u", "--upsample", type=int, default=1,
	help="# of times to upsample")
args = vars(ap.parse_args())

print('--------Face detection with CNN started')

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
boxes = [trim_n_convert(image=img,rectangle=r.rect) for r in rects]

# loop over the bounding boxes
for (x, y, w, h) in boxes:
	# draw the bounding box on our image
	cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
# show the output image
cv2.imshow("Output", img)
cv2.waitKey(5000)