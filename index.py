#Imports
import json
import os
import cv2 as cv

class Haar_cascade_ai:
    def __init__(self, type_id):
        with open(os.path.join('Json', 'typelist.json')) as list_json:
            self.type_list = json.load(list_json)
        self.XML = self.type_list[type_id]
        self.ai = cv.CascadeClassifier(os.path.join('XMLs', self.XML))
    def train(self, data):
        pass
    
    def classfier(self, data):
        pass
    

ai = Haar_cascade_ai(7)
