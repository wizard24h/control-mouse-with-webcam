import cv2
import numpy as np
from keras.models import load_model


# if we get the same class for <accuracy> number of iterations:
#   we set the current class to that class
#   else we return the old current class
class Geture(object):
    accuracy = 0
    counter = 0
    right_hand = False
    prev_class = 'mouse'
    model = load_model("./CNN_model/model_1.h5")
    model._make_predict_function()
    classes = ['click', 'five', 'four', 'hold', 'mouse', 'phone', 'slider', 'three', 'two', 'volume']
    min_points = [0, 3, 3, 1, 1, 2, 1, 2, 1, 0]

    def __init__(self, accuracy = 0):
        self.accuracy = accuracy
        self.counter = accuracy

    def classify(self, roi, num_of_points = 10):
        roi = np.expand_dims(roi, axis=-1)
        roi = np.expand_dims(roi, axis=0)
        prediction = self.model.predict(roi)[0]
        index = np.argmax(prediction)
        if self.min_points[index] > num_of_points:
            print("ERROR!")
            return None
        pred = self.classes[index]

        if not self.apply_accuracy(pred):
            return None
        return pred

    def apply_accuracy(self, gesture):
        if self.prev_class == gesture:
            if self.counter > 0:
                self.counter -= 1
            else:
                return True
        else:
            self.prev_class = gesture
            self.counter = self.accuracy
        return False
