import cv2
import streamlit as st
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os
from pygame import mixer

MIN_CONF = 0.3
NMS_THRESH = 0.3
MIN_DISTANCE = 50

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args(["--input","pedestrians.mp4","--output","my_output.avi","--display","1"]))

labelsPath = os.path.sep.join([r"C:\Users\Justin\OneDrive\Desktop\social distancing alarm\coco.names.txt"])
LABELS = open(labelsPath).read().strip().split("\n")


weightsPath = os.path.sep.join([r"C:\Users\Justin\OneDrive\Desktop\social distancing alarm\yolov3.weights"])
configPath = os.path.sep.join([r"C:\Users\Justin\OneDrive\Desktop\social distancing alarm\yolov3.cfg.txt"])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

def detect_people(frame,net,ln,personIdx=0):

  (H , W ) = frame.shape[:2]
  results = []
  blob = cv2.dnn.blobFromImage(frame,1/255.0,(416,416,),swapRB=True, crop=False)
  net.setInput(blob)
  layerOutputs= net.forward(ln)

  boxes = []
  centroids = []
  confidences = []

  for output in layerOutputs: 
    for detection in output:
      scores = detection[5:]
      classID = np.argmax(scores)
      confidence = scores[classID]
     
      if classID == personIdx and confidence > MIN_CONF:
         box = detection[0:4] * np.array ([W,H,W,H])
         (centerX, centerY, width, height) = box.astype("int")
 
         x = int(centerX - (width/2))
         y = int(centerY - (height/2))


         boxes.append([x, y, int(width), int(height)])
         centroids.append((centerX, centerY))
         confidences.append(float(confidence))

  idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

  if len(idxs) > 0:

    for i in idxs.flatten():

         (x, y) = (boxes[i][0], boxes[i][1])
         (w, h) = (boxes[i][2], boxes[i][3])
         r = (confidences[i], (x, y, x + w, y + h), centroids[i])
	     
         results.append(r)


  return results


st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
writer = None


while run:
    _, frame = camera.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # FRAME_WINDOW.image(frame)
    frame = imutils.resize(frame, width=700)
    results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
    violate = set()
    
    if len(results) >= 2:
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")
        
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                if D[i, j] < MIN_DISTANCE:
                    mixer.init()
                    mixer.music.load('beep1.mp3')
                    mixer.music.play()
                    violate.add(i)
                    violate.add(j)
    
    for (i, (prob, bbox, centroid)) in enumerate(results):
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)
        
        if i in violate:
            color = (0, 0, 255)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)
    
    text = "Social Distancing Violations: {}".format(len(violate))
    cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
    
    if args["display"] > 0:
        cv2.imshow('', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    
    if args["output"] != "" and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25, (frame.shape[1], frame.shape[0]), True)
        
    if writer is not None:
        writer.write(frame)
 
else:
    st.write('Stopped')
