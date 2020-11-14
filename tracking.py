import numpy as np
import cv2
from config import MIN_CONF, NMS_THRESH 

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

def detect_human (net, ln, frame, encoder, tracker, time):
# Get the dimension of the frame
	(frame_height, frame_width) = frame.shape[:2]
	# Initialize lists needed for detection
	boxes = []
	centroids = []
	confidences = []

	# Construct a blob from the input frame 
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)

	# Perform forward pass of YOLOv3, output are the boxes and probabilities
	net.setInput(blob)
	layer_outputs = net.forward(ln)

	# For each output
	for output in layer_outputs:
		# For each detection in output 
		for detection in output:
			# Extract the class ID and confidence 
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			# Class ID for person is 0, check if the confidence meet threshold
			if class_id == 0 and confidence > MIN_CONF:
				# Scale the bounding box coordinates back to the size of the image
				box = detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
				(center_x, center_y, width, height) = box.astype("int")
				# Derive the coordinates for the top left corner of the bounding box
				x = int(center_x - (width / 2))
				y = int(center_y - (height / 2))
				# Add processed results to respective list
				boxes.append([x, y, int(width), int(height)])
				centroids.append((center_x, center_y))
				confidences.append(float(confidence))
	# Perform Non-maxima suppression to suppress weak and overlapping boxes
	# It will filter out unnecessary boxes, i.e. box within box
	# Output will be indexs of useful boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

	tracked_bboxes = []
	expired = []
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
		expired = tracker.update(detections, time)


		# Obtain info from the tracks
		for track in tracker.tracks:
				if not track.is_confirmed() or track.time_since_update > 5:
						continue 
				tracked_bboxes.append(track)

	return [tracked_bboxes, expired]

