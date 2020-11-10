import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import json
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

data_length = len(human_count)

time_axis = []
graph_height = max(human_count)

fig, ax = plt.subplots()
for i in range(data_length):
	time_axis.append(i * data_record_frame)
	if restricted_entry[i]:
		ax.add_patch(patches.Rectangle((i * data_record_frame, 0), data_record_frame, graph_height / 10, facecolor = 'red', fill=True))
	if abnormal_activity[i]:
		ax.add_patch(patches.Rectangle((i * data_record_frame, 0), data_record_frame, graph_height / 20, facecolor = 'blue', fill=True))

violate_line, = plt.plot(time_axis, violate_count, linewidth=3, label="Violation Count")
crowd_line, =  plt.plot(time_axis, human_count, linewidth=3, label="Crowd Count")
plt.title("Crowd Data versus Time")
plt.xlabel("Frames")
plt.ylabel("Count")
re_legend = patches.Patch(color= "red", label="Restriced Entry Detected")
an_legend = patches.Patch(color= "blue", label="Abnormal Crowd Activity Detected")
plt.legend(handles=[crowd_line, violate_line, re_legend, an_legend])
plt.show()	
