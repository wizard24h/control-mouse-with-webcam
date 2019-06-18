from pynput.mouse import Button, Controller, Listener
from datetime import datetime
import os
import numpy as np
import win32api
d = Controller()
prev_mouse_button = 'up'
prev_gesture = 'mouse'
hold = False
time = 0
# cursor mass from 1 to 10
# cursor force is always 1
# we did this to avoid floating points

cursor_mass = 1
sensitivity = 1
frame_shape = (480,640,3)
command_counter = 10

hand1 = None
hand2 = None
hand3 = None
hand4 = None

def gesture_to_controller(point, gesture):
    global prev_gesture, command_counter, accuracy, hold, hand1, hand2, hand3, hand4
    if gesture is None or point is None:
        return
    #fix right and left directions
    center = np.array([frame_shape[0]/2,frame_shape[1]/2])
    point = np.array([frame_shape[0]-point[0],point[1]])
    derivative = point - center
    # print(derivative)
    # print(gesture)
    if gesture == 'mouse':
        mouse_up(derivative)
        prev_gesture = 'mouse'
        command_counter = 10
        return

    # if not apply_accuracy(gesture):
    #     return

    if gesture == 'five':
        mouse_hold(derivative)
        prev_gesture = 'five'
        command_counter = 10
        return
    else:
        if prev_mouse_button == 'hold':
            d.release(Button.left)

    if gesture == 'hold':
        hold = not hold
        prev_gesture = 'hold'
        command_counter = 10
        return
    if hold:
        return

    if gesture == 'click':
        mouse_down(derivative)
        prev_gesture = 'click'
        command_counter = 10
        return

    if gesture == 'slider':
        scroll_horizontal(-derivative[0])
        prev_gesture = 'slider'
        command_counter = 10
        return

    if gesture == 'volume':
        scroll_vertical(-derivative[1])
        prev_gesture = 'volume'
        command_counter = 10
        return

    if gesture == 'two':
        if command_counter > 0:
            command_counter -= 1
            return
        if hand1 is not None and prev_gesture != 'command':
            os.system(hand1)
            prev_gesture = 'command'
            command_counter = 10
    if gesture == 'three':
        if command_counter > 0:
            command_counter -= 1
            return
        if hand2 is not None and prev_gesture != 'command':
            os.system(hand2)
            prev_gesture = 'command'
            command_counter = 10

    if gesture == 'four':
        if command_counter > 0:
            command_counter -= 1
            return
        if hand3 is not None and prev_gesture != 'command':
            os.system(hand3)
            prev_gesture = 'command'
            command_counter = 10

    if gesture == 'phone':
        if command_counter > 0:
            command_counter -= 1
            return
        if hand4 is not None and prev_gesture != 'command':
            os.system(hand4)
            prev_gesture = 'command'
            command_counter = 10


    # print('direction', direction)
# def apply_accuracy(gesture):
#     global prev_gesture, counter, accuracy
#     if prev_gesture == gesture:
#         if counter > 0:
#             counter -= 1
#         else:
#             return True
#     else:
#         prev_gesture = gesture
#         counter = accuracy
#     return False

def mouse_down(derivative):
    global prev_mouse_button
    if prev_mouse_button == 'up':
        d.press(Button.left)
        d.release(Button.left)
        prev_mouse_button = 'down'
    mouse_cursor(derivative)

def mouse_hold(derivative):
    global prev_mouse_button
    if prev_mouse_button == 'up':
        d.press(Button.left)
    prev_mouse_button = 'hold'
    mouse_cursor(derivative)


def mouse_up(derivative):
    global prev_mouse_button
    prev_mouse_button = 'up'
    mouse_cursor(derivative)

def mouse_cursor(derivative):
    old_cursor = d.position
    global time, sensitivity
    if time == 0:
        time = float(datetime.now().strftime('%S.%f'))
    dt = float(datetime.now().strftime('%S.%f')) - time
    time = float(datetime.now().strftime('%S.%f'))
    global cursor_mass
    # velocity = (f/m)*dt.seconds
    if dt > 0.5:
        dt = 0.5
    velocity = (0.5 / cursor_mass) * dt
    # print("velocity: ", velocity)
    # print(derivative)
    derivative *= sensitivity
    movement = derivative * velocity
    # d.position = old_cursor + (derivative * velocity)
    if movement[0] < 20 * sensitivity and movement[1] < 20 * sensitivity:
        new_cursor = np.array( old_cursor + movement)
        d.position= new_cursor.astype(int)
    else:
        print(movement)


def scroll_vertical(direction):
    d.scroll(0, direction)

def scroll_horizontal(direction):
    d.scroll(direction,0)
#
# def volume_up(self):
#
#
# def volume_down(self):
#
#
# def left(self):
#
# def right(self):


