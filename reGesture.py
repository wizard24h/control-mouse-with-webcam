
# import the necessary packages
import numpy as np
import cv2
import glob
import PalmLib
import windowsController
import keyboard
import threading
from DMImage import DMImagePreprocessor
import win32api as win

# import matplotlib
# from matplotlib import pyplot as plt


camera = cv2.VideoCapture(0)
play = False

def slider_changed(slider_name, value):
    if slider_name == 'sensitivity':
        windowsController.sensitivity = int(value)
    if slider_name == 'accuracy':
        PalmLib.classifier.accuracy = int(value)
    if slider_name == 'smoothness':
        windowsController.cursor_mass = int(value)

def button_clicked(name):
    if name == "view_channels":
        PalmLib.toggle_channels()
    if name == "start":
        run()
    if name == "pause":
        global play
        play = False
    if name == 'left_hand':
        PalmLib.hand(False)
    if name == 'right_hand':
        PalmLib.hand(True)
    print('button clicked: '+name)

def available_palms():
    palms_files = glob.glob("./palms/*.palm.npy")
    palms = []
    for i in range(len(palms_files)):
        file_name = palms_files[i].split("\\")[1]
        name = file_name.split(".")[0]
        palms.append(name)
    return palms

def set_shortcut(name,value):
    if name == "hand1":
        windowsController.hand1= value
    if name == "hand2":
        windowsController.hand2= value
    if name == "hand3":
        windowsController.hand3= value
    if name == "hand4":
        windowsController.hand4= value

def create_palm(name):
    global camera
    PalmLib.create_palm(name,camera)

def choose_palm(palm_name):
    new_palm = np.load('./palms/'+palm_name+'.palm.npy')
    PalmLib.activate_palm(new_palm)

def start():
    global camera
    palms_files = glob.glob("./palms/*.palm.npy")
    if len(palms_files) == 0:
        PalmLib.create_palm("main", camera)
    choose_palm("main")

def run():
    global camera, play
    play = True
    frame = camera.read()
    windowsController.frame_shape = frame[1].shape
    while camera.isOpened():
        pressed_key = cv2.waitKey(1)
        _, frame = camera.read()
        point, gesture = PalmLib.palm_info(frame)
        windowsController.gesture_to_controller(point, gesture)

        PalmLib.imshow("Live Feed", PalmLib.rescale_frame(frame))
        if keyboard.is_pressed('q'):
            break
        # if pressed_key & 0xFF == ord('t'):
        #     print(point)
    # cleanup the camera and close any open windows
    # camera.release()
    cv2.destroyAllWindows()
