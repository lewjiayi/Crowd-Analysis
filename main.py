import datetime
import time
import numpy as np
import imutils
import cv2
import time
import os.path as path
import csv
import json
from video_process import video_process
from config import DATA_PRESENT, YOLO_CONFIG, VIDEO_CONFIG, SHOW_PROCESSING_OUTPUT, DATA_RECORD_RATE
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

# Read from video
cap = cv2.VideoCapture(VIDEO_CONFIG["VIDEO_CAP"])
IS_CAM = VIDEO_CONFIG["IS_CAM"]

# Load YOLOv3-tiny weights and config
WEIGHTS_PATH = YOLO_CONFIG["WEIGHTS_PATH"]
CONFIG_PATH = YOLO_CONFIG["CONFIG_PATH"]

# Load the YOLOv3-tiny pre-trained COCO dataset 
net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
# Set the preferable backend to CPU since we are not using GPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get the names of all the layers in the network
ln = net.getLayerNames()
# Filter out the layer names we dont need for YOLO
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Tracker parameters
max_cosine_distance = 0.7
nn_budget = None

#initialize deep sort object
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

movement_data_file = open('movement_data.csv', 'w') 
crowd_data_file = open('crowd_data.csv', 'w')
# sd_violate_data_file = open('sd_violate_data.csv', 'w')
# restricted_entry_data_file = open('restricted_entry_data.csv', 'w')

movement_data_writer = csv.writer(movement_data_file)
crowd_data_writer = csv.writer(crowd_data_file)
# sd_violate_writer = csv.writer(sd_violate_data_file)
# restricted_entry_data_writer = csv.writer(restricted_entry_data_file)

if path.getsize('movement_data.csv') == 0:
	movement_data_writer.writerow(['Track ID', 'Entry time', 'Exit Time', 'Movement Tracks'])
if path.getsize('crowd_data.csv') == 0:
	crowd_data_writer.writerow(['Time', 'Human Count', 'Social Distance violate', 'Restricted Entry'])

VID_FPS = cap.get(cv2.CAP_PROP_FPS)
DATA_RECORD_FRAME = int(VID_FPS / DATA_RECORD_RATE)

# Start counting time for processing speed calculation
t0 = time.time()

frame_count = video_process(cap, net, ln, encoder, tracker, movement_data_writer, crowd_data_writer)
cap.release()
cv2.destroyAllWindows()
movement_data_file.close()
crowd_data_file.close()

# Calculate and print system & processing data
t1 = time.time() - t0
print("Frame Count: ", frame_count)
print("Time elapsed: ", t1)
print("Processed FPS: ", frame_count/t1)

video_data = {
	"IS_CAM": IS_CAM,
	"PROCESSED_FRAMES": frame_count,
	"DATA_RECORD_FRAME" : DATA_RECORD_FRAME,
	"START_TIME": t0,
	"END_TIME": t1
}

with open('video_data.json', 'w') as video_data_file:
	json.dump(video_data, video_data_file)

