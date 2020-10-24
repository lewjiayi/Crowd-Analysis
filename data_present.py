import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import floor

def data_present(frameCount, humanCountFrame, restrictedEntryFrame, violateCountFrame):
	# Plot graphs of violation & crowd count vs time(frame)
	timeAxis = []
	graphHeight = max(humanCountFrame)

	fig, ax = plt.subplots()
	for f in range(floor(frameCount/5)):
		timeAxis.append(f * 5)
		if restrictedEntryFrame[f]:
			# plt.vlines(x = f * 5, ymin = 0, ymax = graphHeight / 5 , colors = "red")
			ax.add_patch(patches.Rectangle((f * 5, 0), 5 , graphHeight / 10, facecolor = 'red', fill=True))

	violateLine, = plt.plot(timeAxis, violateCountFrame, linewidth=3, label="Violation Count")
	crowdLine, =  plt.plot(timeAxis, humanCountFrame, linewidth=3, label="Crowd Count")
	plt.title("Violation & Crowd Count versus Time")
	plt.xlabel("Frames")
	plt.ylabel("Count")
	reLegend = patches.Patch(color= "red", label="Restriced Entry Detected")
	plt.legend(handles=[crowdLine, violateLine, reLegend])
	plt.show()	