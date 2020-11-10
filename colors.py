RGB_COLORS = {
	"blue": (255, 0, 0),
	"green": (0, 255, 0),
	"red": (0, 0, 255),
	"yellow": (0, 255, 255),
	"white": (0, 0, 0),
	"black": (255, 255, 255)
}

def gradient_color_RGB(color1, color2, steps, current):
	step1 = (color2[0] - color1[0])/steps
	step2 = (color2[1] - color1[1])/steps
	step3 = (color2[2] - color1[2])/steps
	color_1 = int(color1[0] + current*step1)
	color_2 = int(color1[1] + current*step2)
	color_3 = int(color1[2] + current*step3)
	return (color_1, color_2, color_3)
	