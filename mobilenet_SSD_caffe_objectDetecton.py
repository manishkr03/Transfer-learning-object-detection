#Import the neccesary libraries
import numpy as np
import argparse
import cv2 

# =============================================================================
# # construct the argument parse 
# parser = argparse.ArgumentParser(
#     description='Script to run MobileNet-SSD object detection network')
# parser.add_argument("--image", default= "img.jpeg", help="path to video file. If empty, camera's stream will be used")
# parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
#                                   help='Path to text network file: '
#                                        'MobileNetSSD_deploy.prototxt for Caffe model'
#                                        )
# parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
#                                  help='Path to weights: '
#                                       'MobileNetSSD_deploy.caffemodel for Caffe model'
#                                       )
# parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
# args = parser.parse_args()
# 
# # Labels of Network.
# classNames = { 0: 'background',
#     1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
#     5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
#     10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
#     14: 'motorbike', 15: 'person', 16: 'pottedplant',
#     17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }
# 
# =============================================================================

thr =0.2
img_pth = "bject_detection\data\living_room.jpg" 
configPath ="object_detectiom_model\MobileNetSSD_deploy.prototxt"
weightsPath = "object_detectiom_model\MobileNetSSD_deploy.caffemodel"

# Labels of Network.
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }
# =============================================================================
# 
# fileNames =r"E:\softweb\Pretrained_model\object_detectiom_model\coco.names"
# 
# classNames =[]
# with open(fileNames, 'r') as names:
#     classNames = names.read().split("\n")
# =============================================================================
    
    
#Load the Caffe model 
net = cv2.dnn.readNetFromCaffe(configPath, weightsPath)
# Load image fro
frame = cv2.imread(img_pth)
frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction
heightFactor = frame.shape[0]/300.0
widthFactor = frame.shape[1]/300.0 
# MobileNet requires fixed dimensions for input image(s)
# so we have to ensure that it is resized to 300x300 pixels.
# set a scale factor to image because network the objects has differents size. 
# We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
# after executing this command our "blob" now has the shape:
# (1, 3, 300, 300)
blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
#Set to network the input blob 
net.setInput(blob)
#Prediction of network
detections = net.forward()

frame_copy = frame.copy()
frame_copy2 = frame.copy()
#Size of frame resize (300x300)
cols = frame_resized.shape[1] 
rows = frame_resized.shape[0]

#For get the class and location of object detected, 
# There is a fix index for class, location and confidence
# value in @detections array .
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2] #Confidence of prediction 
    if confidence > thr: # Filter prediction 
        class_id = int(detections[0, 0, i, 1]) # Class label

        # Object location 
        xLeftBottom = int(detections[0, 0, i, 3] * cols) 
        yLeftBottom = int(detections[0, 0, i, 4] * rows)
        xRightTop   = int(detections[0, 0, i, 5] * cols)
        yRightTop   = int(detections[0, 0, i, 6] * rows)

        xLeftBottom_ = int(widthFactor * xLeftBottom) 
        yLeftBottom_ = int(heightFactor* yLeftBottom)
        xRightTop_   = int(widthFactor * xRightTop)
        yRightTop_   = int(heightFactor * yRightTop)
        # Draw location of object  
        cv2.rectangle(frame_resized, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                      (0, 255, 0))

        cv2.rectangle(frame_copy, (xLeftBottom_, yLeftBottom_), (xRightTop_, yRightTop_),
                      (0, 255, 0),-1)
opacity = 0.3
cv2.addWeighted(frame_copy, opacity, frame, 1 - opacity, 0, frame)

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2] #Confidence of prediction 
    if confidence > thr: # Filter prediction 
        class_id = int(detections[0, 0, i, 1]) # Class label

        # Object location 
        xLeftBottom = int(detections[0, 0, i, 3] * cols) 
        yLeftBottom = int(detections[0, 0, i, 4] * rows)
        xRightTop   = int(detections[0, 0, i, 5] * cols)
        yRightTop   = int(detections[0, 0, i, 6] * rows)

        xLeftBottom_ = int(widthFactor * xLeftBottom) 
        yLeftBottom_ = int(heightFactor* yLeftBottom)
        xRightTop_   = int(widthFactor * xRightTop)
        yRightTop_   = int(heightFactor * yRightTop)
        cv2.rectangle(frame, (xLeftBottom_, yLeftBottom_), (xRightTop_, yRightTop_),
          (0, 0, 0),2)
        # Draw label and confidence of prediction in frame resized
        if class_id in classNames:
            label = classNames[class_id] + ": " + str(confidence)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_TRIPLEX, 0.8, 1)

            yLeftBottom_ = max(yLeftBottom_, labelSize[1])
            cv2.rectangle(frame, (xLeftBottom_, yLeftBottom_ - labelSize[1]),
                                 (xLeftBottom_ + labelSize[0], yLeftBottom_ + baseLine),
                                 (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xLeftBottom_, yLeftBottom_),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 0))
            print (label) #print class and confidence 
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.imshow("frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()



#for video


#Import the neccesary libraries
import numpy as np
import argparse
import cv2 

# construct the argument parse 
# =============================================================================
# parser = argparse.ArgumentParser(
#     description='Script to run MobileNet-SSD object detection network')
# parser.add_argument("--image", default= "img.jpeg", help="path to video file. If empty, camera's stream will be used")
# parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
#                                   help='Path to text network file: '
#                                        'MobileNetSSD_deploy.prototxt for Caffe model'
#                                        )
# parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
#                                  help='Path to weights: '
#                                       'MobileNetSSD_deploy.caffemodel for Caffe model'
#                                       )
# parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
# args = parser.parse_args()
# =============================================================================

# Labels of Network.
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }


thr =0.2

vid_pth = "data\CDC recommends wearing face masks in public.mp4" 

#img_pth = r"E:\softweb\ML\project\object_detection\data\living_room.jpg" 
configPath ="object_detectiom_model\MobileNetSSD_deploy.prototxt"
weightsPath = "object_detectiom_model\MobileNetSSD_deploy.caffemodel"

# =============================================================================
# 
# fileNames =r"E:\softweb\Pretrained_model\object_detectiom_model\coco.names"
# 
# classNames =[]
# with open(fileNames, 'r') as names:
#     classNames = names.read().split("\n")
# =============================================================================
    
    
#Load the Caffe model 
net = cv2.dnn.readNetFromCaffe(configPath, weightsPath)
# Load image fro
cap = cv2.VideoCapture(vid_pth)
# frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction
# heightFactor = frame.shape[0]/300.0
# widthFactor = frame.shape[1]/300.0 
# MobileNet requires fixed dimensions for input image(s)
# so we have to ensure that it is resized to 300x300 pixels.
# set a scale factor to image because network the objects has differents size. 
# We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
# after executing this command our "blob" now has the shape:
# (1, 3, 300, 300)

while True:
    
    ret, frame =cap.read()
    
    frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction
    heightFactor = frame.shape[0]/300.0
    widthFactor = frame.shape[1]/300.0
    
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    #Set to network the input blob 
    net.setInput(blob)
    #Prediction of network
    detections = net.forward()
    
    frame_copy = frame.copy()
    frame_copy2 = frame.copy()
    #Size of frame resize (300x300)
    cols = frame_resized.shape[1] 
    rows = frame_resized.shape[0]
    
    #For get the class and location of object detected, 
    # There is a fix index for class, location and confidence
    # value in @detections array .
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2] #Confidence of prediction 
        if confidence > thr: # Filter prediction 
            class_id = int(detections[0, 0, i, 1]) # Class label
    
            # Object location 
            xLeftBottom = int(detections[0, 0, i, 3] * cols) 
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop   = int(detections[0, 0, i, 5] * cols)
            yRightTop   = int(detections[0, 0, i, 6] * rows)
    
            xLeftBottom_ = int(widthFactor * xLeftBottom) 
            yLeftBottom_ = int(heightFactor* yLeftBottom)
            xRightTop_   = int(widthFactor * xRightTop)
            yRightTop_   = int(heightFactor * yRightTop)
            # Draw location of object  
            cv2.rectangle(frame_resized, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                          (0, 255, 0))
    
            cv2.rectangle(frame_copy, (xLeftBottom_, yLeftBottom_), (xRightTop_, yRightTop_),
                          (0, 255, 0),-1)
    opacity = 0.3
    cv2.addWeighted(frame_copy, opacity, frame, 1 - opacity, 0, frame)
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2] #Confidence of prediction 
        if confidence > thr: # Filter prediction 
            class_id = int(detections[0, 0, i, 1]) # Class label
    
            # Object location 
            xLeftBottom = int(detections[0, 0, i, 3] * cols) 
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop   = int(detections[0, 0, i, 5] * cols)
            yRightTop   = int(detections[0, 0, i, 6] * rows)
    
            xLeftBottom_ = int(widthFactor * xLeftBottom) 
            yLeftBottom_ = int(heightFactor* yLeftBottom)
            xRightTop_   = int(widthFactor * xRightTop)
            yRightTop_   = int(heightFactor * yRightTop)
            cv2.rectangle(frame, (xLeftBottom_, yLeftBottom_), (xRightTop_, yRightTop_),
              (0, 0, 0),2)
            # Draw label and confidence of prediction in frame resized
            if class_id in classNames:
                label = classNames[class_id] + ": " + str(confidence)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_TRIPLEX, 0.8, 1)
    
                yLeftBottom_ = max(yLeftBottom_, labelSize[1])
                cv2.rectangle(frame, (xLeftBottom_, yLeftBottom_ - labelSize[1]),
                                     (xLeftBottom_ + labelSize[0], yLeftBottom_ + baseLine),
                                     (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xLeftBottom_, yLeftBottom_),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 0))
                print (label) #print class and confidence 
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
     # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  
    
cap.release()
cv2.destroyAllWindows()


# =============================================================================
# fileNames ="object_detectiom_model\coco.names"
# 
# clsNames =[]
# with open(fileNames, 'r') as names:
#     clsNames = names.read().split("\n")
#     
#     
# classNames={}
# for n, i in enumerate(clsNames):
#     classNames[n] = i
# 
# print(classNames)
# =============================================================================
    


##########################################################################3
#SIMPLE  MOBILENET SSD OBJECT DETCTION CAFFE MODEL OBJECT DETCTION VIDEO------22
# USAGE
# python real_time_object_detection.py

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

vid_pth = "data\CDC recommends wearing face masks in public.mp4" 

#img_pth = r"E:\softweb\ML\project\object_detection\data\living_room.jpg" 
configPath ="object_detectiom_model\MobileNetSSD_deploy.prototxt"
weightsPath = "object_detectiom_model\MobileNetSSD_deploy.caffemodel"


# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(configPath, weightsPath)

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
cap = cv2.VideoCapture(vid_pth)
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	ret, frame = cap.read()
	#frame = imutils.resize(frame, width=400)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > 0.2:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()





