import numpy as np
import cv2
from config import MIN_CONF, NMS_THRESH 

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

def detect_human (net, ln, frame, encoder, tracker):
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
	# It will filter out unnecessary boxes, i.e. box within box
	# Output will be indexs of useful boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

	tracked_bboxes = []
	if len(idxs) > 0:
		del_idxs = []
		for i in range(len(boxes)):
			if i not in idxs:
				del_idxs.append(i)
		for i in sorted(del_idxs, reverse=True):
			del boxes[i]
			del centroids[i]
			del confidences[i]

		boxes = np.array(boxes)
		centroids = np.array(centroids)
		confidences = np.array(confidences)
		features = np.array(encoder(frame, boxes))
		detections = [Detection(bbox, score, centroid, feature) for bbox, score, centroid, feature in zip(boxes, scores, centroids, features)]

		tracker.predict()
		tracker.update(detections)

		# Obtain info from the tracks
		for track in tracker.tracks:
				if not track.is_confirmed() or track.time_since_update > 5:
						continue 
				bbox = track.to_tlbr() # Get the corrected/predicted bounding box
				tracking_id = track.track_id # Get the ID for the particular track
				tracked_bboxes.append(bbox.tolist() + [tracking_id]) # Structure data, that we could use it with our draw_bbox function

	return tracked_bboxes

