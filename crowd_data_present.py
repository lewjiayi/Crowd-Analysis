import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import csv
import json
import datetime
from math import floor

human_count = []
violate_count = []
restricted_entry = []
abnormal_activity = []
with open('processed_data/crowd_data.csv', 'r') as file:
	reader = csv.reader(file, delimiter=',')
	next(reader)
	for row in reader:
		human_count.append(int(row[1]))
		violate_count.append(int(row[2]))
		restricted_entry.append(bool(int(row[3])))
		abnormal_activity.append(bool(int(row[4])))

with open('processed_data/video_data.json', 'r') as file:
	data = json.load(file)
	data_record_frame = data["DATA_RECORD_FRAME"]
	is_cam = data["IS_CAM"]
	vid_fps = data["VID_FPS"]
	start_time = data["START_TIME"]

start_time= datetime.datetime.strptime(start_time, "%d/%m/%Y, %H:%M:%S")
time_steps = data_record_frame/vid_fps
data_length = len(human_count)

time_axis = []
graph_height = max(human_count)

fig, ax = plt.subplots()
time = start_time
for i in range(data_length):
	time += datetime.timedelta(seconds= time_steps)
	time_axis.append(time)
	next_time = time + datetime.timedelta(seconds= time_steps)
	rect_width = mdates.date2num(next_time) - mdates.date2num(time)
	if restricted_entry[i]:
		ax.add_patch(patches.Rectangle((mdates.date2num(time), 0), rect_width, graph_height / 10, facecolor = 'red', fill=True))
	if abnormal_activity[i]:
		ax.add_patch(patches.Rectangle((mdates.date2num(time), 0), rect_width, graph_height / 20, facecolor = 'blue', fill=True))


violate_line, = plt.plot(time_axis, violate_count, linewidth=3, label="Violation Count")
crowd_line, =  plt.plot(time_axis, human_count, linewidth=3, label="Crowd Count")
plt.title("Crowd Data versus Time")
plt.xlabel("Time")
plt.ylabel("Count")
re_legend = patches.Patch(color= "red", label="Restriced Entry Detected")
an_legend = patches.Patch(color= "blue", label="Abnormal Crowd Activity Detected")
plt.legend(handles=[crowd_line, violate_line, re_legend, an_legend])
plt.show()	
