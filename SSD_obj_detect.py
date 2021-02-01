#for images

import cv2
import imutils
thres = 0.40

fileNames ="object_detectiom_model\coco.names"

with open(fileNames, 'r') as names:
    classNames = names.read().split("\n")
    
img_pth = "data\living_room.jpg" 
configPath ="object_detectiom_model\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "bject_detectiom_model\frozen_inference_graph.pb"


net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


img =cv2.imread(img_pth)
img = imutils.resize(img, width =800)


classIds, confs, bbox = net.detect(img, confThreshold=thres, nms_threshold =0.)

for clsid, confidence, boxes in zip(classIds.flatten(), confs.flatten(), bbox):
    
    Label = '{:0.2f}'.format(float(confidence))
    Label = "{}%".format(float(Label)*100)
    label = "{} :{}".format(classNames[clsid-1], Label)
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1 )
    #(left, top) and (right, bottom)
    #left, top, right, bottom = boxes
    #cv2.rectangle(img, boxes, (0,255,0), 2)
    x, y, w, h = boxes
    top = max(y, labelSize[1])
    cv2.rectangle(img, (x,y),(x+w, y+h) , (0,255,0), 2)
    cv2.rectangle(img, (x, top - labelSize[1]), (x + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(img,label, (x,top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()



##########################################################
#for videos

import cv2
import imutils
thres = 0.40

vid_pth = "data\CDC recommends wearing face masks in public.mp4" 

cap = cv2.VideoCapture(vid_pth)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)


fileNames ="object_detectiom_model\coco.names"

classNames =[]
with open(fileNames, 'r') as names:
    classNames = names.read().split("\n")
    
configPath ="object_detectiom_model\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "object_detectiom_model\frozen_inference_graph.pb"


net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)



while True:
    ret, img =cap.read()
    
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    print(classIds)
    if len(classIds) != 0:

        for clsid, confidence, boxes in zip(classIds.flatten(), confs.flatten(), bbox):
            
            Label = '{:0.2f}'.format(float(confidence))
            Label = "{}%".format(float(Label)*100) 
            label = "{}:{}".format(classNames[clsid-1], Label)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1 )
            #(left, top) and (right, bottom)
            #left, top, right, bottom = boxes
            #cv2.rectangle(img, boxes, (0,255,0), 2)
            x, y, w, h = boxes
            top = max(y, labelSize[1])
            cv2.rectangle(img, (x,y),(x+w, y+h) , (0,255,0), 2)
            cv2.rectangle(img, (x, top - labelSize[1]), (x + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
            cv2.putText(img,label, (x,top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
    
    cv2.imshow("img", img)
    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  

cap.release()
cv2.destroyAllWindows()
    
    






##for images------ removing Duplicate NMS

import cv2
import imutils
import numpy as np
thres = 0.45

fileNames ="object_detectiom_model\coco.names"

with open(fileNames, 'r') as names:
    classNames = names.read().split("\n")
    
img_pth = "object_detection\data\living_room.jpg" 
configPath ="object_detectiom_model\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "object_detectiom_model\frozen_inference_graph.pb"


net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


img =cv2.imread(img_pth)
img = imutils.resize(img, width =800)


classIds, confs, bbox = net.detect(img, confThreshold=thres)

bbox = list(bbox)
confs = list(np.array(confs).reshape(1,-1)[0])
confs = list(map(float,confs))
#print(type(confs[0]))
#print(confs)
nms_indices =cv2.dnn.NMSBoxes(bbox, confs, thres,nms_threshold =0.2)
print(nms_indices)

for i in nms_indices:
    i =i[0]
    #print(i)
    nms_bbox = bbox[i]
    nms_classIds = classIds[i][0]
    nms_confs = confs[i]
    print(nms_confs)
    #print(nms_clsIds)
    #print(classNames[i])
    #print(nms_box)
    Label = '{:0.2f}'.format(float(nms_confs))
    Label = "{}%".format(float(Label)*100)
    label = "{} :{}".format(classNames[nms_classIds-1], Label)
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1 )
    #(left, top) and (right, bottom)
    #left, top, right, bottom = boxes
    #cv2.rectangle(img, boxes, (0,255,0), 2)
    x, y, w, h = nms_bbox
    top = max(y, labelSize[1])
    cv2.rectangle(img, (x,y),(x+w, y+h) , (0,255,0), 2)
    cv2.rectangle(img, (x, top - labelSize[1]), (x + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(img,label, (x,top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
    




    
##for video------ removing Duplicate NMS

import cv2
import imutils
import numpy as np
thres = 0.45


vid_pth = "data\CDC recommends wearing face masks in public.mp4" 

cap = cv2.VideoCapture(vid_pth)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

fileNames ="object_detectiom_model\coco.names"

with open(fileNames, 'r') as names:
    classNames = names.read().split("\n")
    
#img_pth = "data\living_room.jpg" 
configPath ="object_detectiom_model\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "object_detectiom_model\frozen_inference_graph.pb"


net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    
    ret, img = cap.read()

    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    #print(type(confs[0]))
    #print(confs)
    nms_indices =cv2.dnn.NMSBoxes(bbox, confs, thres,nms_threshold =0.2)
    #print(nms_indices)
    
    if len(classIds) != 0:
        for i in nms_indices:
            i =i[0]
            #print(i)
            nms_bbox = bbox[i]
            nms_classIds = classIds[i][0]
            nms_confs = confs[i]
            print(nms_confs)
            #print(nms_clsIds)
            #print(classNames[i])
            #print(nms_box)
            Label = '{:0.2f}'.format(float(nms_confs))
            Label = "{}%".format(float(Label)*100)
            label = "{} :{}".format(classNames[nms_classIds-1], Label)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1 )
            #(left, top) and (right, bottom)
            #left, top, right, bottom = boxes
            #cv2.rectangle(img, boxes, (0,255,0), 2)
            x, y, w, h = nms_bbox
            top = max(y, labelSize[1])
            cv2.rectangle(img, (x,y),(x+w, y+h) , (0,255,0), 2)
            cv2.rectangle(img, (x, top - labelSize[1]), (x + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
            cv2.putText(img,label, (x,top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  
cap.release()
cv2.destroyAllWindows()



