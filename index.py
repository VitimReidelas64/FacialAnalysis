#Imports
import time
import json
import os
import cv2 as cv


class Webcam:
	def __init__(self):
		self.webcam = cv.VideoCapture(0)
	def read(self):
		return self.webcam.read()



class Haar_cascade_ai:
    def __init__(self, type_id):
        with open(os.path.join('Json', 'typelist.json')) as list_json:
            self.type_list = json.load(list_json)
        self.XML = self.type_list[type_id]
        self.ai = cv.CascadeClassifier(os.path.join('XMLs', self.XML))
    def train(self, type_id):
        with open(os.path.join('Json', 'configs.json')) as list_json:
            self.type_list = json.load(list_json)
        
    
    def locate(self, image):
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return [self.ai.detectMultiScale(img_gray), image, img_gray]


class Drawner:
    def __init__(self):
        pass
    def show(self, img, idents=[], wait=1 * 1000, title="Untitled"):
        for ident in idents:
            self.drawretangles(img, ident['size'], ident['color'])
        cv.imshow(title, img)
        return cv.waitKey(wait)

        
            
    

    def drawretangles(self, img, locs, color):
        for loc in locs:
            cv.rectangle(img, (loc[0], loc[1]), (loc[0] + loc[2], loc[1] + loc[3]), color, 2)

wbcam = Webcam()
dr = Drawner()
ai0 = Haar_cascade_ai(7)
ai1 = Haar_cascade_ai(1)
while True:
    wc = wbcam.read()[1]
    res0 = ai0.locate(wc)
    res1 = ai1.locate(wc)
    key0 = dr.show(wc, wait=5, title="AI", idents=[
        {
            "size": res0[0],
            "color": (0, 255, 0)           
        },
        {
            "size": res1[0],
            "color": (0, 0, 255)
        }
    ])
    key1 = dr.show(res0[2], wait=5, title="IA", idents=[
        {
            "size": res0[0],
            "color": (0, 255, 0)           
        },
        {
            "size": res1[0],
            "color": (0, 0, 255)
        }
    ])
    if key0 == 27 or key1 == 27:#ESC
        break
