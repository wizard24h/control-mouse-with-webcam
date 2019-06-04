# palm tracking and detection library
# created by omar aboulubdeh for multimedia controller project


import cv2
import numpy as np
from DMImage import DMImagePreprocessor


class Palm(object):
    def __init__(self):
        pass

    def hand_histogram(frame):
        global hand_rect_one_x, hand_rect_one_y

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # region of interested
        roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)
        for i in range(total_rectangle):
            roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                              hand_rect_one_y[i]:hand_rect_one_y[i] + 10]
        hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)

    def detect_hsv(self, scale=1):
        camera = cv2.VideoCapture(0)

        while camera.isOpened():
            pressed_key = cv2.waitKey(1)
            _, frame = camera.read()
            if pressed_key & 0xFF == ord('z'):
                is_hand_hist_created = True
                return hand_histogram(frame)

                frame = draw_rect(frame)

            cv2.imshow("Live Feed", rescale_frame(frame))

            if pressed_key == 27:
                break
