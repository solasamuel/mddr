# import the necessary packages
import argparse
import cv2
import numpy as np

# Create a 600x600 px grayscale canvas
canvas = np.ones((600, 600), dtype="uint8") * 255
# Designate a 400x400 px point of interest and set to black
canvas[100:500, 100:500] = 0

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
		# draw a rectangle around the region of interest
		cv2.rectangle(canvas, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", canvas)

# Create named cv2 window
cv2.namedWindow("Test Canvas")
# Register mouse hook callback from on_mouse_events() function
cv2.setMouseCallback("Test Canvas", click_and_crop)


# key presses and their interpretations
while(True):
    cv2.imshow("Canvas", canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): # q to close window & end script
        break
    elif key == ord('c'):
        canvas[100:500, 100:500] = 0
cv2.destroyAllWindows()