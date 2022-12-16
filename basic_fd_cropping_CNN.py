import utiles
import argparse
import imutils
import time
import dlib
import cv2
import os

"""
Module orignially provided to us by project owner. Implementaion of the dlib face detection model.
Module has been customized to fit our needs.

Contributions: Erik Norén and Sushruth Badri
"""

'''import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
dlib.DLIB_USE_CUDA = True'''

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-sf", "--start_folder", type=str, required=True,
	help="path to folder with input images")
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

# assign directory
directory = args["start_folder"]

start_run = time.time()
print('Iterating over files in directory ' + args["start_folder"])

nr_fcs = 0
tot_img = 0
face_count = 0
# iterate over files in
# that directory
for i, filename in enumerate(os.listdir(directory)):
    f = os.path.join(directory, filename)
    
    # checking if it is a file
    if os.path.isfile(f):
        # Child directroy
        #subfolder = os.path.splitext(os.path.basename(f))[0]

        # Parent Directory path
        # TODO: Change name of end_folder
        end_folder = args['end_folder']

        # Path
        path = end_folder
        
        # Create a subfolder for the current image in the end_folder
        #os.mkdir(path)
        '''os.chdir(end_folder)
        os.mkdir(subfolder)
        os.chdir('..')'''
        
        print(f)
        img = cv2.imread(f)
        #img = imutils.resize(img, width=600)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detection
        start = time.time()
        print('Face detection with CNN started on image ' + filename)
        rects = detector(rgb, args["upsample"]) # this is a list
        end = time.time()
        print(f'Run time on image {end-start}')

        # Fix the boxes
        boxes = [utiles.trim_n_convert_v2(image=img, epsilon = 5, rectangle=r.rect) for r in rects]

        # To store images in the desired folder
        os.chdir(path)
        
        
        
        # loop over the bounding boxes
        for (x, y, w, h) in boxes:
            # draw the bounding box on our image
            fcs = img[y:y+h, x:x+w]
            #fcs = imutils.resize(fcs, width=100)
            cv2.imwrite('face'+str(face_count)+'.jpg', fcs)
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_count += 1
            
        #nr_fcs += box_count
        tot_img = i

        # Going back to root directory
        #print(os.getcwd())
        os.chdir('../') # TO BE CHANGED ACCORDINGLY
        
print('Processed ' + str(tot_img+1) + ' images')
print('Detected ' + str(face_count) + ' faces')

end_run = time.time()
print(f'Total runtime {end_run-start_run}')