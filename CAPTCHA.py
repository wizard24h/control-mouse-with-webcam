
# import the necessary packages
from flask import Flask, render_template
import numpy as np
import cv2
import glob
import PalmLib
import windowsController
import datetime, time
import keyboard
import threading
from DMImage import DMImagePreprocessor
import win32api as win

# import matplotlib
# from matplotlib import pyplot as plt

camera = cv2.VideoCapture(0)
play = False
app = Flask(__name__,static_folder='Images', static_url_path='/Images')
def available_palms():
    palms_files = glob.glob("./palms/*.palm.npy")
    palms = []
    for i in range(len(palms_files)):
        file_name = palms_files[i].split("\\")[1]
        name = file_name.split(".")[0]
        palms.append(name)
    return palms

def create_palm(name):
    PalmLib.create_palm(name)

def choose_palm(palm_name):
    new_palm = np.load('./palms/'+palm_name+'.palm.npy')
    PalmLib.activate_palm(new_palm)


@app.route('/captcha/',methods=['GET'])
def run():
    palms_files = glob.glob("./palms/*.palm.npy")
    if len(palms_files) == 0:
        PalmLib.create_palm("main")
    choose_palm("main")

    global camera, play
    play = True
    frame = camera.read()
    windowsController.frame_shape = frame[1].shape
    PalmLib.classifier.accuracy= 25
    t1 = datetime.datetime.now() + datetime.timedelta(seconds=15)
    while camera.isOpened():
        pressed_key = cv2.waitKey(1)
        _, frame = camera.read()
        cv2.putText(frame, "What is the answer of 18-14 ?", (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (82, 234, 201,255), lineType=cv2.LINE_AA )
        cv2.imshow("Live Feed", PalmLib.rescale_frame(frame))
        point, gesture = PalmLib.palm_info(frame)
        if gesture == 'four':
            return "true"
        # elif gesture != None:
        #     return "false"
        if t1 < datetime.datetime.now():
            return "false"
        if gesture is not None:
            return "false"
        # if pressed_key & 0xFF == ord('t'):
        #     print(point)
    # cleanup the camera and close any open windows
    # camera.release()

@app.route('/')
def index():
    return render_template('index2.html')

if __name__ == "__main__":
    app.run(debug=True)

