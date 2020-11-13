import datetime

# Show individuals detected
SHOW_PROCESSING_OUTPUT = True
# Show individuals detected
SHOW_DETECT = True
# Data record
DATA_RECORD = True
# Data record rate (data record per frame)
DATA_RECORD_RATE = 3
# Check for restricted entry
RE_CHECK = True
# Restricted entry time (H:M:S)
RE_START_TIME = datetime.time(0,0,0) 
RE_END_TIME = datetime.time(23,0,0)
# Check for social distance violation
SD_CHECK = True
# Show violation count
SHOW_VIOLATION_COUNT = True
# Threshold for distance violation
SOCIAL_DISTANCE = 50
# Check for abnormal crowd activity
ABNORMAL_CHECK = True
# Min number of people to check for abnormal
ABNORMAL_MIN_PEOPLE = 5
# Abnormal energy level threshold
ABNORMAL_ENERGY = 500
# Abnormal activity ratio threhold
ABNORMAL_THRESH = 0.66
# Threshold for human detection minumun confindence
MIN_CONF = 0.3
# Threshold for Non-maxima surpression
NMS_THRESH = 0.2
# Resize frame for processing
FRAME_SIZE = 720

# Video Path
VIDEO_CONFIG = {
	"VIDEO_CAP" : "video/7.mp4",
	"IS_CAM" : False,
	"HIGH_CAM": False,
	"START_TIME": datetime.datetime(2020, 11, 5, 0, 0, 0, 0)
}

# Load YOLOv3-tiny weights and config
YOLO_CONFIG = {
	"WEIGHTS_PATH" : "YOLOv4-tiny/yolov4-tiny.weights",
	"CONFIG_PATH" : "YOLOv4-tiny/yolov4-tiny.cfg"
}
