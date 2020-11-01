import csv
import imutils
import time
import cv2
from config import VIDEO_CONFIG
from itertools import zip_longest
from math import ceil
from colors import RGB_COLORS, gradient_color_RGB

print("Loading tracks")
t0 = time.time()
tracks = []
with open('movement_data.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        if len(row[3:]) > 4:
            temp = []
            for x in row[3:]:
                temp.append(int(x))
            tracks.append(temp)

t1 = time.time() - t0
print("Time elapsed: ", t1)

cap = cv2.VideoCapture(VIDEO_CONFIG["VIDEO_CAP"])
cap.set(1, 100)
(ret, frame) = cap.read()
frame = imutils.resize(frame, width=720)

print("Drawing tracks")
t0 = time.time()

color1 = RGB_COLORS["green"]
color2 = RGB_COLORS["red"]
for track in tracks:
    length = ceil(len(track)/2)
    for i in range(0, length, 2):
        color = gradient_color_RGB(color1, color2, length, i)
        cv2.line(frame, (track[i], track[i+1]), (track[i+2], track[i+3]), color, 2)

t1 = time.time() - t0
print("Time elapsed: ", t1)

frame = imutils.resize(frame, width=1080)

cv2.imshow("Movement Tracks" ,frame)
cv2.waitKey()
cv2.destroyAllWindows()
cap.release()

    