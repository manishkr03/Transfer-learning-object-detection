#yolo object detection image ---1

import cv2
import numpy as np
import time


confThreshold =0.5
nmsThreshold= 0.2

image_pth = "data\living_room.jpg"
# load the COCO class labels our YOLO model was trained on
labelsPath = 'yolo_weights\coco.names'
# derive the paths to the YOLO weights and model configuration
weightsPath = 'yolo_weights\yolov4.weights'
configPath = 'yolo_weights\yolov4.cfg'

LABELS  =[]
with open(labelsPath, "r") as f:
    LABELS  = f.read().strip("\n").split("\n")
    
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

net =cv2.dnn.readNetFromDarknet(configPath, weightsPath)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

image =cv2.imread(image_pth)
(H, W) = image.shape[:2]

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)

net.setInput(blob)
start = time.time()

layerOutputs = net.forward(ln)
# [,frame,no of detections,[classid,class score,conf,x,y,h,w]
end = time.time()

# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))
# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
boxes = []
confidences = []
classIDs = []

# loop over each of the layer outputs
for output in layerOutputs:
	# loop over each of the detections
	for detection in output:
		# extract the class ID and confidence (i.e., probability) of
		# the current object detection
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > confThreshold:
			# scale the bounding box coordinates back relative to the
			# size of the image, keeping in mind that YOLO actually
			# returns the center (x, y)-coordinates of the bounding
			# box followed by the boxes' width and height
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")

			# use the center (x, y)-coordinates to derive the top and
			# and left corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			# update our list of bounding box coordinates, confidences,
			# and class IDs
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

# ensure at least one detection exists
if len(idxs) > 0:
	# loop over the indexes we are keeping
	for i in idxs.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		# draw a bounding box rectangle and label on the image
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


###########################################################

#yolo object detection video----1

import cv2
import numpy as np
import time


confThreshold =0.5
nmsThreshold= 0.2

vid_pth = "data\CDC recommends wearing face masks in public.mp4" 

#image_pth = r"E:\softweb\ML\project\object_detection\data\living_room.jpg"
# load the COCO class labels our YOLO model was trained on
labelsPath = 'yolo_weights\coco.names'
# derive the paths to the YOLO weights and model configuration
weightsPath = 'yolo_weights\yolov4.weights'
configPath = 'yolo_weights\yolov4.cfg'

LABELS  =[]
with open(labelsPath, "r") as f:
    LABELS  = f.read().strip("\n").split("\n")
    
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

net =cv2.dnn.readNetFromDarknet(configPath, weightsPath)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#image =cv2.imread(image_pth)
#(H, W) = image.shape[:2]
cap =cv2.VideoCapture(vid_pth)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
while True:
    ret, image =cap.read()
    
    if not ret:
        break
    
    (H, W) = image.shape[:2]
    

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
    	swapRB=True, crop=False)
    
    net.setInput(blob)
    start = time.time()
    
    layerOutputs = net.forward(ln)
    # [,frame,no of detections,[classid,class score,conf,x,y,h,w]
    end = time.time()
    
    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    
    # loop over each of the layer outputs
    for output in layerOutputs:
    	# loop over each of the detections
    	for detection in output:
    		# extract the class ID and confidence (i.e., probability) of
    		# the current object detection
    		scores = detection[5:]
    		classID = np.argmax(scores)
    		confidence = scores[classID]
    
    		# filter out weak predictions by ensuring the detected
    		# probability is greater than the minimum probability
    		if confidence > confThreshold:
    			# scale the bounding box coordinates back relative to the
    			# size of the image, keeping in mind that YOLO actually
    			# returns the center (x, y)-coordinates of the bounding
    			# box followed by the boxes' width and height
    			box = detection[0:4] * np.array([W, H, W, H])
    			(centerX, centerY, width, height) = box.astype("int")
    
    			# use the center (x, y)-coordinates to derive the top and
    			# and left corner of the bounding box
    			x = int(centerX - (width / 2))
    			y = int(centerY - (height / 2))
    
    			# update our list of bounding box coordinates, confidences,
    			# and class IDs
    			boxes.append([x, y, int(width), int(height)])
    			confidences.append(float(confidence))
    			classIDs.append(classID)
    
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    
    # ensure at least one detection exists
    if len(idxs) > 0:
    	# loop over the indexes we are keeping
    	for i in idxs.flatten():
    		# extract the bounding box coordinates
    		(x, y) = (boxes[i][0], boxes[i][1])
    		(w, h) = (boxes[i][2], boxes[i][3])
    
    		# draw a bounding box rectangle and label on the image
    		color = [int(c) for c in COLORS[classIDs[i]]]
    		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
    		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
    			0.5, color, 2)
    
    # show the output image
    cv2.imshow("Image", image)
    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  
cap.release()
cv2.destroyAllWindows()




##########################################################
#simple yolo image object detction----2


import cv2
import numpy as np
 

#vid_pth = r"E:\softweb\ML\project\object_detection\data\CDC recommends wearing face masks in public.mp4" 

image_pth = "object_detection\data\living_room.jpg"
# load the COCO class labels our YOLO model was trained on
classesFile = 'yolo_weights\coco.names'
# derive the paths to the YOLO weights and model configuration
modelWeights = 'yolo_weights\yolov3.weights'
modelConfiguration = 'yolo_weights\yolov3.cfg'

#cap  = cv2.VideoCapture(vid_pth)
img =cv2.imread(image_pth)
whT = 320
confThreshold =0.5
nmsThreshold= 0.2
 
#### LOAD MODEL
## Coco Names
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)
## Model Files

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
 

blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
net.setInput(blob)
layersNames = net.getLayerNames()
outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
outputs = net.forward(outputNames)


import pandas as pd
df =pd.DataFrame(outputs)


hT, wT, cT = img.shape
bbox = []
classIds = []
confs = []
for output in outputs:
    #print(output)
    for det in output:
        #print(det)
        scores = det[5:]
        #print(scores)
        classId = np.argmax(scores)
        confidence = scores[classId]
        #print(confidence)
        if confidence > confThreshold:
            w,h = int(det[2]*wT) , int(det[3]*hT)
            #print(det[3])
            x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
            bbox.append([x,y,w,h])
            classIds.append(classId)
            confs.append(float(confidence))
 
indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

if len(indices) > 0:
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        cv2.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
 

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()






#simple yolo video object detction---2


import cv2
import numpy as np
 

vid_pth = "data\CDC recommends wearing face masks in public.mp4" 

#image_pth = r"E:\softweb\ML\project\object_detection\data\living_room.jpg"
# load the COCO class labels our YOLO model was trained on
classesFile = 'yolo_weights\coco.names'
# derive the paths to the YOLO weights and model configuration
modelWeights = 'yolo_weights\yolov3.weights'
modelConfiguration = 'yolo_weights\yolov3.cfg'

cap  = cv2.VideoCapture(vid_pth)
#img =cv2.imread(image_pth)
whT = 320
confThreshold =0.5
nmsThreshold= 0.2
 
#### LOAD MODEL
## Coco Names
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)
## Model Files

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
 
while True:
    ret, img = cap.read()
    
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    
    
    import pandas as pd
    df =pd.DataFrame(outputs)
    
    
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        #print(output)
        for det in output:
            #print(det)
            scores = det[5:]
            #print(scores)
            classId = np.argmax(scores)
            confidence = scores[classId]
            #print(confidence)
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                #print(det[3])
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
     
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    
    if len(indices) > 0:
        for i in indices:
            i = i[0]
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            # print(x,y,w,h)
            cv2.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2)
            cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                      (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
     
    
    cv2.imshow('Image', img)
    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  
cap.release()
cv2.destroyAllWindows()




