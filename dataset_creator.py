# import the necessary packages
import numpy as np
import cv2
import glob
import PalmLib

from DMImage import DMImagePreprocessor
import win32api as win

# import matplotlib
# from matplotlib import pyplot as plt
cursor = []
def create_palm(name):
    PalmLib.create_palm(name)
def choose_palm(palm_name):
    new_palm = np.load('./palms/'+palm_name+'.palm.npy')
    PalmLib.activate_palm(new_palm)

def run():
    palms_files = glob.glob("./palms/*.palm.npy")
    if len(palms_files) == 0:
        PalmLib.create_palm("main")
    choose_palm("main")
    camera = cv2.VideoCapture(0)
    instanceCount = 0
    while camera.isOpened():
        pressed_key = cv2.waitKey(1)
        _, frame = camera.read()
        mask, cont = PalmLib.palm_info(frame)
        x,y,w,h = cv2.boundingRect(cont)
        p1 = (x, y)
        p2 = (x+w, y+h)
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 1)
        PalmLib.imshow("Live Feed", PalmLib.rescale_frame(frame))
        roi = mask[y:y+h,x:x+w]
        dataset = PalmLib.roi_to_square(roi)
        instanceCount +=1
        if instanceCount <= 200:
            path = "./dataset/train/volume/" + str(instanceCount) + '.png'
            cv2.imwrite(path, dataset)
        else :
            path = "./dataset/test/volume/" + str(instanceCount) + '.png'
            cv2.imwrite(path, dataset)

        if instanceCount == 300:
            break
        if pressed_key == 27:
            break
    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    run()
