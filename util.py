from scipy.spatial.distance import euclidean

# Calculate shortest distance between two rectangle
def rect_distance(rect1, rect2):
	(x1, y1, x1b, y1b) = rect1
	(x2, y2, x2b, y2b) = rect2
	# Rect 2 is at the left of rect 1
	left = x2b < x1
	# Rect 2 is at the right of rect 1
	right = x1b < x2
	# Rect 2 is at the bottom of rect 1
	bottom = y2b < y1
	# Rect 2 is at the top of rect 1
	top = y1b < y2
	if top and left:
		return euclidean((x1, y1b), (x2b, y2))
	elif left and bottom:
		return euclidean((x1, y1), (x2b, y2b))
	elif bottom and right:
		return euclidean((x1b, y1), (x2, y2b))
	elif right and top:
		return euclidean((x1b, y1b), (x2, y2))
	elif left:
		return x1 - x2b
	elif right:
		return x2 - x1b
	elif bottom:
		return y1 - y2b
	elif top:
		return y2 - y1b
	else:
		# Rect 1 & 2 intersects
		return  0

def progress(frame_count):
	import sys
	sys.stdout.write('\r')
	if frame_count % 2 == 0:
		sys.stdout.write("Processing .. ")
	else:
		sys.stdout.write("Processing .  ")
	sys.stdout.flush()

def kinetic_energy(point1, point2, time_step):
	speed = euclidean(point1, point2) / time_step
	return int(0.5 * speed ** 2)