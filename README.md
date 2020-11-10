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
