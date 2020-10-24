import datetime
import numpy as np
import imutils
import cv2
import time
from math import ceil
from tracking import detect_human
from data_present import data_present
from util import rect_distance, progress
from colors import RGB_COLORS
from config import SHOW_DETECT, DATA_PRESENT, RE_CHECK, RE_START_TIME, RE_END_TIME, SD_CHECK, SOCIAL_DISTANCE, SHOW_PROCESSING_OUTPUT, YOLO_CONFIG, VIDEO_CONFIG
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
IS_CAM = VIDEO_CONFIG["IS_CAM"]

def video_process(cap, net, ln, encoder, tracker, data_input):
	frame_count = 0
	violate_count_frame = []
	violate_period_total = 0
	human_count_frame = []
	human_period_total = 0
	restricted_entry_frame = []
	restricted_entry_period = False
	re_warning_timeout = 0
	sd_warning_timeout = 0
	vid_frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	while True:
		(ret, frame) = cap.read()

		# Stop the loop when video ends
		if not ret:
			break

		# Resize Frame to 720p
		frame = imutils.resize(frame, width=720)
		if frame_count > 1000000:
			frame_count = 0
		frame_count += 1

		# Get current time
		current_datetime = datetime.datetime.now()

		# Run detection algorithm
		if IS_CAM:
			record_time = current_datetime
		else:
			record_time = frame_count
		[humans_detected, expired] = detect_human(net, ln, frame, encoder, tracker, record_time)
		humans_detected = [list(map(int, detect)) for detect in humans_detected]


		violate = set()
		if SD_CHECK:
			# Initialize set for violate so an individual will be recorded only once
			violate_count = np.zeros(len(humans_detected))
			# Check the distance between all combinations of detection
			if len(humans_detected) >= 2:
				for i in range (0, len(humans_detected)):
					for j in range (i + 1, len(humans_detected)):
						if rect_distance(humans_detected[i][:4], humans_detected[j][:4]) < SOCIAL_DISTANCE:
							# Distance between detection less than minimum social distance 
							violate.add(i)
							violate_count[i] += 1
							violate.add(j)
							violate_count[j] += 1
		
		# Check for restricted entry
		RE = False
		if RE_CHECK:
			if (current_datetime.time() > RE_START_TIME) and (current_datetime.time() < RE_END_TIME) :
				if len(humans_detected) > 0:
					RE = True
			
		for i, human in enumerate(humans_detected):
			[x, y, w, h, cx, cy, id] = human
			if i in violate:
				# cv2.putText((frame), str(int(violate_count[i])), (x	, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RGB_COLORS["yellow"], 2)
				cv2.rectangle(frame, (x, y), (w, h), RGB_COLORS["yellow"], 2)
				cv2.circle(frame, (cx, cy), 5, RGB_COLORS["yellow"], 2)
				cv2.putText(frame, str(id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RGB_COLORS["yellow"], 2)
			elif SHOW_DETECT and not RE:
				cv2.rectangle(frame, (x , y), (w, h), RGB_COLORS["green"], 2)
				cv2.putText(frame, str(id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RGB_COLORS["green"], 2)
				cv2.circle(frame, (cx, cy), 5, RGB_COLORS["green"], 2)
			if RE:
				cv2.rectangle(frame, (x - 5 , y - 5 ), (w + 5, h + 5), RGB_COLORS["red"], 5)

		if SD_CHECK:
			# Warning stays on screen for 10 frames
			if (len(violate) > 0):
				sd_warning_timeout = 10
			else: 
				sd_warning_timeout -= 1
			# Display violation warning and count on screen
			if sd_warning_timeout > 0:
				text = "Violation count: {}".format(len(violate))
				cv2.putText(frame, text, (200, frame.shape[0] - 30),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
		
		if RE_CHECK:
			# Warning stays on screen for 10 frames
			if RE:
				re_warning_timeout = 10
			else: 
				re_warning_timeout -= 1
			# Display restricted entry warning and count on screen
			if re_warning_timeout > 0:
				if frame_count % 3 != 0 :
					cv2.putText(frame, "RESTRICTED ENTRY", (200, 200),
						cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

		if SHOW_DETECT:
			# Display crowd count on screen
			text = "Crowd count: {}".format(len(humans_detected))
			cv2.putText(frame, text, (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

		current_date = str(current_datetime.strftime("%b-%d-%Y"))
		current_time = str(current_datetime.strftime("%I:%M:%S %p"))
		cv2.putText(frame, (current_date), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
		cv2.putText(frame, (current_time), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
		
		# Store data violation and crowd count data for graph plotting
		# Data store on average of every 5 frames
		if DATA_PRESENT:
			if frame_count % 5 == 0:
				violate_count_frame.append(ceil(violate_period_total / 5))
				human_count_frame.append(ceil(human_period_total / 5))
				violate_period_total = 0
				human_period_total = 0
				restricted_entry_frame.append(restricted_entry_period)
				restricted_entry_period = False
			else:
				violate_period_total += len(violate)
				human_period_total += len(humans_detected)
				if RE:
					restricted_entry_period = True

		if SHOW_PROCESSING_OUTPUT:
			cv2.imshow("Processed Output", frame)
		else:
			progress(frame_count, vid_frame_length)

		# cv2.waitKey()
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
	cv2.destroyAllWindows()
	data_input.send([frame_count, human_count_frame, restricted_entry_frame, violate_count_frame])
	data_input.close()