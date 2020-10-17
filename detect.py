import numpy as np
import cv2
from config import MIN_CONF, NMS_THRESH 

def detect_human (net, ln, frame):
# Get the dimension of the frame
	(frameHeight, frameWidth) = frame.shape[:2]
	# Initialize lists needed for detection
	boxes = []
	centroids = []
	confidences = []

	# Construct a blob from the input frame 
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)

	# Perform forward pass of YOLOv3, output are the boxes and probabilities
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	# For each output
	for output in layerOutputs:
		# For each detection in output 
		for detection in output:
			# Extract the class ID and confidence 
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			# Class ID for person is 0, check if the confidence meet threshold
			if classID == 0 and confidence > MIN_CONF:
				# Scale the bounding box coordinates back to the size of the image
				box = detection[0:4] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])
				(centerX, centerY, width, height) = box.astype("int")
				# Derive the coordinates for the top left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				# Add processed results to respective list
				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))
	# Perform Non-maxima suppression to suppress weak and overlapping boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

	humansDetected = []
	# Check if there are detection
	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# Add probability, coordinates and centroid to detection list
			r = (confidences[i], (x, y, x + w, y + h), centroids[i])
			humansDetected.append(r)

	return humansDetected