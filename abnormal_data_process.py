import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import csv
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

tracks = []
with open('processed_data/movement_data.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        if len(row[3:]) > 4:
            temp = []
            data = row[3:]
            for i in range(0, len(data), 2):
                temp.append([int(data[i]), int(data[i+1])])
            tracks.append(temp)

with open('processed_data/video_data.json', 'r') as file:
	data = json.load(file)
	data_record_frame = data["DATA_RECORD_FRAME"]
	frame_size = data["PROCESSED_FRAME_SIZE"]
	vid_fps = data["VID_FPS"]

time_steps = data_record_frame/vid_fps
stationary_distance = frame_size * 0.01

print("Tracks recorded: " + str(len(tracks)))

useful_tracks = []
for movement in tracks:
	stationary = movement[0]
	track = [stationary]
	for i in movement[1:]:
		if euclidean(stationary, i) > stationary_distance:
			stationary = i
			track.append(i)
	if len(track) > 1 :
		useful_tracks.append(track)

energies = []
for movement in useful_tracks:
    for i in range(len(movement) - 1):
        speed = round(euclidean(movement[i], movement[i+1]) / time_steps , 2)
        energies.append(int(0.5 * speed ** 2))

c = len(energies)
print("Useful movement data: " + str(c))

energies = pd.Series(energies)
energies = energies[abs(energies - np.mean(energies)) < 2 * np.std(energies)]
x = { 'Energy': energies}
df = pd.DataFrame(x)
print("Outliers removed: " + str(c - df.Energy.count()))
print()
print("Summary of processed data")
print(df.describe())
print()
print("Acceptable energy level (mean value ** 1.05) is " + str(int(df.Energy.mean() ** 1.05)))

bins = np.linspace(int(min(energies)), int(max(energies)),100) 
plt.xlim([min(energies)-5, max(energies)+5])
plt.hist(energies, bins=bins, alpha=0.5)
plt.title('Distribution of energies level')
plt.xlabel('Energy level')
plt.ylabel('Count')

plt.show()
