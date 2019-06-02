# import the necessary packages
# from pyimagesearch import imutils
import numpy as np
import argparse
import matplotlib
from matplotlib import pyplot as plt

import cv2

hand_hist = None
traverse_point = []
total_rectangle = 9
hand_rect_one_x = None
hand_rect_one_y = None

hand_rect_two_x = None
hand_rect_two_y = None


def draw_rect(frame):
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    x = int(6 * rows / 20)
    y = int(9 * cols / 20)
    # cv2.imshow('test',frame[x:x+10, y:y+10, :])
    hsv = cv2.cvtColor(frame[x:x+10, y:y+10, :], cv2.COLOR_BGR2HSV)
    cv2.imshow('test',hsv)

    print(hsv[5,5])
    for i in range(total_rectangle):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)

    return frame

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
args = vars(ap.parse_args())

# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
lower1 = np.array([120, 0, 0], dtype="uint8")
upper1 = np.array([180, 130, 255], dtype="uint8")

lower2 = np.array([0, 4, 0], dtype="uint8")
upper2 = np.array([20, 90, 255], dtype="uint8")

# if a video path was not supplied, grab the reference
# to the gray
if not args.get("video", False):
    camera = cv2.VideoCapture(0)

# otherwise, load the video
else:
    camera = cv2.VideoCapture(args["video"])

# keep looping over the frames in the video
# grab the current frame

frame = cv2.imread('img.png')
# (grabbed, frame) = camera.read()
# if we are viewing a video and we did not grab a
# frame, then we have reached the end of the video
# if args.get("video") and not grabbed:
# break

rows, cols, _ =frame.shape
# frame = draw_rect(frame)
converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# plt.imshow(converted);
# plt.show()
# cv2.imshow('test',converted)
mask1 = cv2.inRange(converted, lower1, upper1)
mask2 = cv2.inRange(converted, lower2, upper2)
skinMask = np.bitwise_or(mask1, mask2)
#cv2.imshow('test1',skinMask)
# cv2.imshow('test2',mask2);

# apply a series of erosions and dilations to the mask
# using an elliptical kernel
K1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
K2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
cv2.imshow('test2', skinMask)

skinMask = cv2.erode(skinMask, K2, iterations=2)
skinMask = cv2.dilate(skinMask, K1, iterations=4)

# blur the mask to help remove noise, then apply the
# mask to the frame
skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
cv2.imshow('test1',skinMask);

skin = cv2.bitwise_and(frame, frame, mask=skinMask)

cv2.imshow("images", np.hstack([frame, skin]))

# show the skin in the image along with the mask
# cv2.imshow("images", np.hstack([frame, skin]))
# if the 'q' key is pressed, stop the loop
while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()