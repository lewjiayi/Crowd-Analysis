import datetime
import time
import numpy as np
import imutils
import cv2
import time
import os.path as path
import csv
from data_present import data_present
from video_process import video_process
from config import DATA_PRESENT, YOLO_CONFIG, VIDEO_CONFIG, SHOW_PROCESSING_OUTPUT
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

file = open('movement_data.csv', 'w') 
writer = csv.writer(file)
if path.getsize('movement_data.csv') == 0:
	writer.writerow(['Track ID', 'Entry time', 'Exit Time', 'Movement Tracks'])

# Start counting time for processing speed calculation
t0 = time.time()

[frame_count, human_count_frame, restricted_entry_frame, violate_count_frame] = video_process(cap, net, ln, encoder, tracker, writer)
cap.release()
cv2.destroyAllWindows()
file.close()

# Calculate and print system & processing data
t1 = time.time() - t0
print("Frame Count: ", frame_count)
print("Time elapsed: ", t1)
print("Processed FPS: ", frame_count/t1)

if DATA_PRESENT:
	data_present(frame_count, human_count_frame, restricted_entry_frame, violate_count_frame)
