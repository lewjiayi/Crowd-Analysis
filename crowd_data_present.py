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
with open('processed_data/crowd_data.csv', 'r') as file:
	reader = csv.reader(file, delimiter=',')
	next(reader)
	for row in reader:
		human_count.append(int(row[1]))
		violate_count.append(int(row[2]))
		restricted_entry.append(bool(row[3]))

with open('processed_data/video_data.json', 'r') as file:
	data = json.load(file)
	frame_count = data["PROCESSED_FRAMES"]
	data_record_frame = data["DATA_RECORD_FRAME"]

data_length = len(human_count)

timeAxis = []
graphHeight = max(human_count)

fig, ax = plt.subplots()
for i in range(data_length):
	timeAxis.append(i * data_record_frame)
	if restricted_entry:
		ax.add_patch(patches.Rectangle((i * data_record_frame, 0), data_record_frame, graphHeight / 10, facecolor = 'red', fill=True))

violateLine, = plt.plot(timeAxis, violate_count, linewidth=3, label="Violation Count")
crowdLine, =  plt.plot(timeAxis, human_count, linewidth=3, label="Crowd Count")
plt.title("Violation & Crowd Count versus Time")
plt.xlabel("Frames")
plt.ylabel("Count")
reLegend = patches.Patch(color= "red", label="Restriced Entry Detected")
plt.legend(handles=[crowdLine, violateLine, reLegend])
plt.show()	