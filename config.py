import datetime

# Show individuals detected
SHOW_DETECT = True
# Data presentation
DATA_PRESENT = False
# Check for restricted entry
RE_CHECK = True
# Restricted entry time
RE_START_TIME = datetime.time(0,0,0) 
RE_END_TIME = datetime.time(23,0,0)
# Check for social distance violation
SD_CHECK = True
# Threshold for distance violation
SOCIAL_DISTANCE = 50
# Threshold for human detection minumun confindence
MIN_CONF = 0.3
# Threshold for Non-maxima surpression
NMS_THRESH = 0.2