import cv2
import numpy as np
import os
import PIL
import glob
import tensorflow
import tkinter as tk
from tensorflow import keras
from PIL import Image, ImageDraw, ImageGrab
from keras.models import load_model

# load model
print("Loading neural network model. . .")
model = load_model(r'C:\\Users\\sholz\\Documents\\repos\\mddr\\model.h5')
print("Load operation successful")

print("Loading GUI . . .")
# create window
root = tk.Tk()
root.resizable(0,0)
root.title("Mouse/Handwritten Digit Recognition")

# initialize variables
lastx, lasty = None, None
image_number = 0

# set a canvas
cv = tk.Canvas(root, width=640, heigh=480, bg='white')
cv.grid(row=0, column=0, pady=2, sticky="W", columnspan=2)

# define functions
# to clear canvas
def clear_screen():
    global cv
    cv.delete("all")

# to set the stage
def activate_event(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y

def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=8, fill='black', capstyle='round', smooth='true', splinesteps=12)
    lastx, lasty = x, y

def predict_digit():
    global image_number
    predictions = []
    percentage =[]
    filename = f'predictions/{image_number}.png'        # TODO: add time and date stamp
    widget = cv

    # get widget coordinates
    x = root.winfo_rootx() + widget.winfo_x()
    y = root.winfo_rooty() + widget.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()

    # grab the drawn image and save it
    ImageGrab.grab().crop((x, y, x1, y1)).save(filename)

    # read the image in color
    image = cv2.imread(filename, cv2.IMREAD_COLOR)

    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply Otsu thresholding
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # extract contours
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for cnt in contours:
        # get bounding box and extract ROI
        x, y, w, h = cv2.boundingRect(cnt)

        # create rectangle
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0 , 0), 1)
        top = int(0.05 * th.shape[0])
        bottom = top
        left = int(0.05 * th.shape[1])
        right = left
        th_up = cv2.copyMakeBorder(th, top , bottom, left, right, cv2.BORDER_REPLICATE)

        # extract the image ROI
        roi = th[y-top: y+h+bottom, x-left: x+w+right]

        # resize roi image to 28x28 pixels
        img = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

        # reshape the image
        img = img.reshape(1, 28, 28, 1)

        # normalize the image
        img = img/255.0

        # predict the result
        pred = model.predict([img])[0]
        final_pred = np.argmax(pred)
        data = str(final_pred) + '  ' + str(int(max(pred)*100)) + '%'

        # draw predic on image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(image, data, (x, y-5), font, font_scale, color, thickness)


    # show results in window
    cv2.imshow('Processed Image', image)
    cv2.waitKey(0)


cv.bind('<Button-1>', activate_event)

# set buttons and labels
button_predict = tk.Button(text="Predict Digit", command=predict_digit)
button_predict.grid(row=2, column=0, padx=10, pady=10)
button_clear = tk.Button(text="Clear Screen", command=clear_screen)
button_clear.grid(row=2, column=1, padx=10, pady=10)

# run application
root.mainloop()
