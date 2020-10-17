from math import ceil, floor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import datetime
import numpy as np
import imutils
import cv2
import time
from detect import detect_human
from util import rect_distance
from config import SHOW_DETECT, DATA_PRESENT, RE_CHECK, RE_START_TIME, RE_END_TIME, SD_CHECK, SOCIAL_DISTANCE

RGB_COLORS = {
	"green": (0, 255, 0),
	"red": (255, 0, 0),
	"yellow": (0, 255, 255),
	"white": (0, 0, 0),
	"black": (255, 255, 255)
}

# Read from video
cap = cv2.VideoCapture("video/5.mp4")

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

	# Run detection algorithm
	humansDetected = detect_human(net, ln, frame)

	violate = set()
	if SD_CHECK:
		# Initialize set for violate so an individual will be recorded only once
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
		(startX, startY, endX, endY) = bbox
		if i in violate:
			cv2.putText((frame), str(int(violateCount[i])), (startX	, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RGB_COLORS["yellow"], 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), RGB_COLORS["yellow"], 2)
		elif SHOW_DETECT and not RE:
			cv2.rectangle(frame, (startX , startY), (endX, endY), RGB_COLORS["green"], 2)

		if RE:
			cv2.rectangle(frame, (startX - 5 , startY - 5 ), (endX + 5, endY + 5), (0, 0, 255), 5)
		
	if SD_CHECK:
		# Warning stays on screen for 10 frames
		if (len(violate) > 0):
			sdWarningTimeout = 10
		else: 
			sdWarningTimeout -= 1
		# Display violation warning and count on screen
		if sdWarningTimeout > 0:
			text = "Violation count: {}".format(len(violate))
			cv2.putText(frame, text, (200, frame.shape[0] - 30),
				cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
	
	if RE_CHECK:
		# Warning stays on screen for 10 frames
		if RE:
			reWarningTimeout = 10
		else: 
			reWarningTimeout -= 1
		# Display restricted entry warning and count on screen
		if reWarningTimeout > 0:
			if frameCount % 3 != 0 :
				cv2.putText(frame, "RESTRICTED ENTRY", (200, 200),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

	if SHOW_DETECT:
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
	if cv2.waitKey() & 0xFF == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()

# Calculate and print system & processing data
t1 = time.time() - t0
print("Frame Count: ", frameCount)
print("Time elapsed: ", t1)
print("Processed FPS: ", frameCount/t1)

if DATA_PRESENT:
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