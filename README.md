# Crowd-Analysis

**Detailed documentation will be updated soon! Stay tuned**

The project is dedicated to apply on CCTV system for crowd analysis. Human detection is implemented using YOLOv4 via OpenCV built-in function. Tracking algorithm is implemented using Deep SORT, referencing the implementation by [Python Lessons](https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3). 

---

## Functions

Current function implementation includes:

- Social distance rule violation
- Entries to restriced areas
- Abnormal crowd movement/activity
- Crowd movement tracks and flow
- Crowd stationaries point (Heatmap)

## Sample Demo (for now)

**Optical flow of crowd movement**
![Optical flow](assets/optical%20flow.png)

**Stationary location Heatmap**
![Heatmap](assets/heatmap.png)

**Detection & Tracking**
![Detection & Tracking](assets/detection.png)

**Social distance violation**
![Social distance violation](assets/social%20distance.png)

**Video summary**
![Video Summary](assets/crowd%20data.png)

## Files needed

To run the code you will need to download the YOLO weights and cfg. Create a folder ```YOLOv4-tiny```, download the file [yolov4-tiny.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights) and [yolov4-tiny.cfg](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-tiny.cfg). You can also use YOLOv4 instead, just replace with the desired weights and cfg 