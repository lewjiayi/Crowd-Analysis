import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import csv
import json
import numpy as np
import pandas as pd
from math import ceil
from scipy.spatial.distance import euclidean

with open('processed_data/video_data.json', 'r') as file:
    data = json.load(file)
    data_record_frame = data["DATA_RECORD_FRAME"]
    frame_size = data["PROCESSED_FRAME_SIZE"]
    vid_fps = data["VID_FPS"]
    track_max_age = data["TRACK_MAX_AGE"]

track_max_age = 3
time_steps = data_record_frame/vid_fps
stationary_time = ceil(track_max_age / time_steps)
stationary_distance = frame_size * 0.01


tracks = []
with open('processed_data/movement_data.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        if len(row[3:]) > stationary_time * 2:
            temp = []
            data = row[3:]
            for i in range(0, len(data), 2):
                temp.append([int(data[i]), int(data[i+1])])
            tracks.append(temp)

print("Tracks recorded: " + str(len(tracks)))

useful_tracks = []
for movement in tracks:
    check_index = stationary_time
    start_point = 0
    track = movement[:check_index]
    while check_index < len(movement):
        for i in movement[check_index:]:
            if euclidean(movement[start_point], i) > stationary_distance:
                track.append(i)
                start_point += 1
                check_index += 1
            else:
                start_point += 1
                check_index += 1
                break
        useful_tracks.append(track)
        track = movement[start_point:check_index]

energies = []
for movement in useful_tracks:
    for i in range(len(movement) - 1):
        speed = round(euclidean(movement[i], movement[i+1]) / time_steps , 2)
        energy = int(0.5 * speed ** 2)
        energies.append(energy)

c = len(energies)
print()
print("Useful movement data: " + str(c))

energies = pd.Series(energies)
x = { 'Energy': energies}
df = pd.DataFrame(x)
print("Kurtosis: " + str(df.kurtosis()[0]))
print("Skew: " + str(df.skew()[0]))
print("Summary of processed data")
print(df.describe())
print("Acceptable energy level (mean value ** 1.05) is " + str(int(df.Energy.mean() ** 1.05)))
bins = np.linspace(int(min(energies)), int(max(energies)),100) 
plt.xlim([min(energies)-5, max(energies)+5])
plt.hist(energies, bins=bins, alpha=0.5)
plt.title('Distribution of energies level')
plt.xlabel('Energy level')
plt.ylabel('Count')

plt.show()

while df.skew()[0] > 7.5:
    print()
    c = len(energies)
    print("Useful movement data: " + str(c))
    energies = energies[abs(energies - np.mean(energies)) < 3 * np.std(energies)]
    x = { 'Energy': energies}
    df = pd.DataFrame(x)
    print("Outliers removed: " + str(c - df.Energy.count()))
    print("Kurtosis: " + str(df.kurtosis()[0]))
    print("Skew: " + str(df.skew()[0]))
    print("Summary of processed data")
    print(df.describe())
    print("Acceptable energy level (mean value ** 1.05) is " + str(int(df.Energy.mean() ** 1.05)))

    bins = np.linspace(int(min(energies)), int(max(energies)),100) 
    plt.xlim([min(energies)-5, max(energies)+5])
    plt.hist(energies, bins=bins, alpha=0.5)
    plt.title('Distribution of energies level')
    plt.xlabel('Energy level')
    plt.ylabel('Count')

    plt.show()
