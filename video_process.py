import datetime
import numpy as np
import imutils
import cv2
import time
from math import ceil
from tracking import detect_human
from util import rect_distance, progress, kinetic_energy
from colors import RGB_COLORS
from config import SHOW_DETECT, DATA_RECORD, RE_CHECK, RE_START_TIME, RE_END_TIME, SD_CHECK, SHOW_VIOLATION_COUNT, SOCIAL_DISTANCE,\
	SHOW_PROCESSING_OUTPUT, YOLO_CONFIG, VIDEO_CONFIG, DATA_RECORD_RATE, ABNORMAL_CHECK, ABNORMAL_ENERGY, ABNORMAL_THRESH, ABNORMAL_MIN_PEOPLE
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
IS_CAM = VIDEO_CONFIG["IS_CAM"]

def _record_movement_data(movement_data_writer, movement):
	track_id = movement.track_id 
	entry_time = movement.entry 
	exit_time = movement.exit			
	positions = movement.positions
	positions = np.array(positions).flatten()
	positions = list(positions)
	data = [track_id] + [entry_time] + [exit_time] + positions
	movement_data_writer.writerow(data)

def _record_crowd_data(time, human_count, violate_count, restricted_entry, abnormal_activity, crowd_data_writer):
	data = [time, human_count, violate_count, int(restricted_entry), int(abnormal_activity)]
	crowd_data_writer.writerow(data)

def _end_video(tracker, frame_count, movement_data_writer):
	for t in tracker.tracks:
		if t.is_confirmed():
			t.exit = frame_count
			_record_movement_data(movement_data_writer, t)
		

def video_process(cap, frame_size, net, ln, encoder, tracker, movement_data_writer, crowd_data_writer):
	VID_FRAME_LENGTH = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	VID_FPS = cap.get(cv2.CAP_PROP_FPS)
	DATA_RECORD_FRAME = int(VID_FPS / DATA_RECORD_RATE)
	TIME_STEP = DATA_RECORD_FRAME/VID_FPS

	frame_count = 0
	display_frame_count = 0
	re_warning_timeout = 0
	sd_warning_timeout = 0
	ab_warning_timeout = 0

	RE = False
	ABNORMAL = False

	while True:
		(ret, frame) = cap.read()

		# Stop the loop when video ends
		if not ret:
			_end_video(tracker, frame_count, movement_data_writer)
			break

		# Update frame count
		if frame_count > 1000000:
			frame_count = 0
			display_frame_count = 0
		frame_count += 1
		
		# Skip frames according to given rate
		if frame_count % DATA_RECORD_FRAME != 0:
			continue

		display_frame_count += 1

		# Resize Frame to given size
		frame = imutils.resize(frame, width=frame_size)

		# Get current time
		current_datetime = datetime.datetime.now()

		# Run detection algorithm
		if IS_CAM:
			record_time = current_datetime
		else:
			record_time = frame_count
		
		# Run tracking algorithm
		[humans_detected, expired] = detect_human(net, ln, frame, encoder, tracker, DATA_RECORD_FRAME, record_time)
		humans_detected = [list(map(int, detect)) for detect in humans_detected]

		for movement in expired:
			_record_movement_data(movement_data_writer, movement)

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
		if RE_CHECK:
			RE = False
			if (current_datetime.time() > RE_START_TIME) and (current_datetime.time() < RE_END_TIME) :
				if len(humans_detected) > 0:
					RE = True
			
		if ABNORMAL_CHECK:
			usable_tracks = 0
			abnormal_tracks = 0
			ABNORMAL = False
			for track in tracker.tracks:
				if track.time_since_update == 0 and track.is_confirmed():
					usable_tracks += 1
					ke = kinetic_energy(track.positions[-1], track.positions[-2], TIME_STEP)
					if ke > ABNORMAL_ENERGY:
						abnormal_tracks += 1
			if usable_tracks > ABNORMAL_MIN_PEOPLE:
				if abnormal_tracks / usable_tracks > ABNORMAL_THRESH:
					ABNORMAL = True
		
		if SHOW_PROCESSING_OUTPUT and (SHOW_DETECT or SD_CHECK or RE_CHECK):
			for i, human in enumerate(humans_detected):
				[x, y, w, h, cx, cy, id] = human
				if i in violate:
					cv2.rectangle(frame, (x, y), (w, h), RGB_COLORS["yellow"], 2)
					if SHOW_VIOLATION_COUNT:
						cv2.putText(frame, str(int(violate_count[i])), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RGB_COLORS["yellow"], 2)
				elif SHOW_DETECT and not RE:
					cv2.rectangle(frame, (x , y), (w, h), RGB_COLORS["green"], 2)
					if SHOW_VIOLATION_COUNT:
						cv2.putText(frame, str(int(violate_count[i])), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RGB_COLORS["green"], 2)
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
				if display_frame_count % 3 != 0 :
					cv2.putText(frame, "RESTRICTED ENTRY", (200, 100),
						cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

		if ABNORMAL_CHECK:
			if ABNORMAL:
				ab_warning_timeout = 10
			else:
				ab_warning_timeout -= 1
			if ab_warning_timeout > 0:
				if display_frame_count % 3 != 0:
					cv2.putText(frame, "ABNORMAL ACTIVITY", (130, 250),
						cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5)

		if SHOW_DETECT:
			# Display crowd count on screen
			text = "Crowd count: {}".format(len(humans_detected))
			cv2.putText(frame, text, (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

		current_date = str(current_datetime.strftime("%b-%d-%Y"))
		current_time = str(current_datetime.strftime("%I:%M:%S %p"))
		cv2.putText(frame, (current_date), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
		cv2.putText(frame, (current_time), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
		

		if DATA_RECORD:
			_record_crowd_data(record_time, len(humans_detected), len(violate), RE, ABNORMAL, crowd_data_writer)

		if SHOW_PROCESSING_OUTPUT:
			cv2.imshow("Processed Output", frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			_end_video(tracker, frame_count, movement_data_writer)
			break
	
	cv2.destroyAllWindows()
	return frame_count
