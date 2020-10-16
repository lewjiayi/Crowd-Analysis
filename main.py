from scipy.spatial.distance import euclidean
from math import ceil, floor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import datetime
import numpy as np
import imutils
import cv2
import time


##################### Modify values below for testing (GUI not implemented) #####################
# Read from video
cap = cv2.VideoCapture("video/5.mp4")
# Check for restricted entry
RE_CHECK = True
# Restricted entry time
RE_START_TIME = datetime.time(0,0,0) 
RE_END_TIME = datetime.time(23,0,0)
# Threshold for distance violation
SOCIAL_DISTANCE = 50
# Threshold for human detection minumun confindence
MIN_CONF = 0.3
# Threshold for Non-maxima surpression
NMS_THRESH = 0.2
##################### Modify values above for testing (GUI not implemented) #####################

# Calculate shortest distance between two rectangle
def rect_distance(rect1, rect2):
	(x1, y1, x1b, y1b) = rect1
	(x2, y2, x2b, y2b) = rect2
	# Rect 2 is at the left of rect 1
	left = x2b < x1
	# Rect 2 is at the right of rect 1
	right = x1b < x2
	# Rect 2 is at the bottom of rect 1
	bottom = y2b < y1
	# Rect 2 is at the top of rect 1
	top = y1b < y2
	if top and left:
		return euclidean((x1, y1b), (x2b, y2))
	elif left and bottom:
		return euclidean((x1, y1), (x2b, y2b))
	elif bottom and right:
		return euclidean((x1b, y1), (x2, y2b))
	elif right and top:
		return euclidean((x1b, y1b), (x2, y2))
	elif left:
		return x1 - x2b
	elif right:
		return x2 - x1b
	elif bottom:
		return y1 - y2b
	elif top:
		return y2 - y1b
	else:
		# Rect 1 & 2 intersects
		return  (SOCIAL_DISTANCE - 1)

# Load YOLOv3-tiny weights and config
weightsPath = "YOLOv3-tiny/yolov3-tiny.weights"
configPath = "YOLOv3-tiny/yolov3-tiny.cfg"

# Load the YOLOv3-tiny pre-trained COCO dataset 
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
# Set the preferable backend to CPU since we are not using GPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get the names of all the layers in the network
ln = net.getLayerNames()
# Filter out the layer names we dont need for YOLO
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Start counting time for processing speed calculation
t0 = time.time()

# Initialize values and list needed
frameCount = 0
violateCountFrame = []
violatePeriodTotal = 0
humanCountFrame = []
humanPeriodTotal = 0
restrictedEntryFrame = []
restrictedEntryPeriod = False
reWarningTimeout = 0
sdWarningTimeout = 0

while True:
	(ret, frame) = cap.read()

	# Stop the loop when video ends
	if not ret:
		break

	# Resize Frame to 720p
	frame = imutils.resize(frame, width=720)
	frameCount += 1
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
	violate = set()
	# Check if there are detection
	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# Add probability, coordinates and centroid to detection list
			r = (confidences[i], (x, y, x + w, y + h), centroids[i])
			humansDetected.append(r)

		# Initialize set for violate so only individual will be recorded only once even violate more than once 
		violateCount = np.zeros(len(humansDetected))
		# Check the distance between all combinations of detection
		if len(humansDetected) >= 2:
			for i in range (0, len(humansDetected)):
				for j in range (i + 1, len(humansDetected)):
					if rect_distance(humansDetected[i][1], humansDetected[j][1]) < SOCIAL_DISTANCE:
						# Distance between detection less than minimum social distance 
						violate.add(i)
						violateCount[i] += 1
						violate.add(j)
						violateCount[j] += 1

	# Get current time
	currentDateTime = datetime.datetime.now()
	# Check for restricted entry
	RE = False
	if RE_CHECK:
		if (currentDateTime.time() > RE_START_TIME) and (currentDateTime.time() < RE_END_TIME) :
			if len(humansDetected) > 0:
				RE = True

	# Draw boxes for each detection green for normal, red for violation detected
	for (i, (prob, bbox, centroid)) in enumerate(humansDetected):
		color = (0, 255, 0)
		(startX, startY, endX, endY) = bbox
		if i in violate:
			color = (0, 255, 255)
			cv2.putText((frame), str(int(violateCount[i])), (startX	, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
			if RE:
				cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

		if RE:
			cv2.rectangle(frame, (startX - 5 , startY - 5 ), (endX + 5, endY + 5), (0, 0, 255), 5)
		else:
			cv2.rectangle(frame, (startX - 3 , startY - 3 ), (endX + 3, endY + 3), color, 2)

	# Warning stays on screen for 10 frames
	if (len(violate) > 0):
		sdWarningTimeout = 10
	else: 
		sdWarningTimeout -= 1
	if RE:
		reWarningTimeout = 10
	else: 
		reWarningTimeout -= 1

	# Display violation warning and count on screen
	if sdWarningTimeout > 0:
		text = "Violation count: {}".format(len(violate))
		cv2.putText(frame, text, (200, frame.shape[0] - 30),
			cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
	# Display restricted entry warning and count on screen
	if reWarningTimeout > 0:
		if frameCount % 3 != 0 :
			cv2.putText(frame, "RESTRICTED ENTRY", (200, 200),
				cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

	# Display crowd count on screen
	text = "Crowd count: {}".format(len(humansDetected))
	cv2.putText(frame, text, (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

	currentDate = str(currentDateTime.strftime("%b-%d-%Y"))
	currentTime = str(currentDateTime.strftime("%I:%M:%S %p"))
	cv2.putText(frame, (currentDate), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
	cv2.putText(frame, (currentTime), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
	
	# Store data violation and crowd count data for graph plotting
	# Data store on average of every 5 frames
	if frameCount % 5 == 0:
		violateCountFrame.append(ceil(violatePeriodTotal / 5))
		humanCountFrame.append(ceil(humanPeriodTotal / 5))
		violatePeriodTotal = 0
		humanPeriodTotal = 0
		restrictedEntryFrame.append(restrictedEntryPeriod)
		restrictedEntryPeriod = False
	else:
		violatePeriodTotal += len(violate)
		humanPeriodTotal += len(humansDetected)
		if RE:
			restrictedEntryPeriod = True
	
	cv2.imshow("Processed Output", frame)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()

# Calculate and print system & processing data
t1 = time.time() - t0
print("Frame Count: ", frameCount)
print("Time elapsed: ", t1)
print("Processed FPS: ", frameCount/t1)

# Plot graphs of violation & crowd count vs time(frame)
timeAxis = []
graphHeight = max(humanCountFrame)

fig, ax = plt.subplots()
for f in range(floor(frameCount/5)):
	timeAxis.append(f * 5)
	if restrictedEntryFrame[f]:
		# plt.vlines(x = f * 5, ymin = 0, ymax = graphHeight / 5 , colors = "red")
		ax.add_patch(patches.Rectangle((f * 5, 0), 5 , graphHeight / 10, facecolor = 'red', fill=True))

violateLine, = plt.plot(timeAxis, violateCountFrame, linewidth=3, label="Violation Count")
crowdLine, =  plt.plot(timeAxis, humanCountFrame, linewidth=3, label="Crowd Count")
plt.title("Violation & Crowd Count versus Time")
plt.xlabel("Frames")
plt.ylabel("Count")
reLegend = patches.Patch(color= "red", label="Restriced Entry Detected")
plt.legend(handles=[crowdLine, violateLine, reLegend])
plt.show()