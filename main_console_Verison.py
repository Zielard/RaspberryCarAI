#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import cv2
from flask import Flask, render_template, Response, request
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
import matplotlib.pyplot as plt
#from PIL import Image, ImageDraw, ImageFont
import time
import six.moves.urllib as urllib
import os
import tarfile
import imutils

import threading

from six import BytesIO

# Load the COCO Label Map
elapsed = []
category_index = {
    1: {'id': 1, 'name': 'person'},
    2: {'id': 2, 'name': 'bicycle'},
    3: {'id': 3, 'name': 'car'},
    4: {'id': 4, 'name': 'motorcycle'},
    5: {'id': 5, 'name': 'airplane'},
    6: {'id': 6, 'name': 'bus'},
    7: {'id': 7, 'name': 'train'},
    8: {'id': 8, 'name': 'truck'},
    9: {'id': 9, 'name': 'boat'},
    10: {'id': 10, 'name': 'traffic light'},
    11: {'id': 11, 'name': 'fire hydrant'},
    13: {'id': 13, 'name': 'stop sign'},
    14: {'id': 14, 'name': 'parking meter'},
    15: {'id': 15, 'name': 'bench'},
    16: {'id': 16, 'name': 'bird'},
    17: {'id': 17, 'name': 'cat'},
    18: {'id': 18, 'name': 'dog'},
    19: {'id': 19, 'name': 'horse'},
    20: {'id': 20, 'name': 'sheep'},
    21: {'id': 21, 'name': 'cow'},
    22: {'id': 22, 'name': 'elephant'},
    23: {'id': 23, 'name': 'bear'},
    24: {'id': 24, 'name': 'zebra'},
    25: {'id': 25, 'name': 'giraffe'},
    27: {'id': 27, 'name': 'backpack'},
    28: {'id': 28, 'name': 'umbrella'},
    31: {'id': 31, 'name': 'handbag'},
    32: {'id': 32, 'name': 'tie'},
    33: {'id': 33, 'name': 'suitcase'},
    34: {'id': 34, 'name': 'frisbee'},
    35: {'id': 35, 'name': 'skis'},
    36: {'id': 36, 'name': 'snowboard'},
    37: {'id': 37, 'name': 'sports ball'},
    38: {'id': 38, 'name': 'kite'},
    39: {'id': 39, 'name': 'baseball bat'},
    40: {'id': 40, 'name': 'baseball glove'},
    41: {'id': 41, 'name': 'skateboard'},
    42: {'id': 42, 'name': 'surfboard'},
    43: {'id': 43, 'name': 'tennis racket'},
    44: {'id': 44, 'name': 'bottle'},
    46: {'id': 46, 'name': 'wine glass'},
    47: {'id': 47, 'name': 'cup'},
    48: {'id': 48, 'name': 'fork'},
    49: {'id': 49, 'name': 'knife'},
    50: {'id': 50, 'name': 'spoon'},
    51: {'id': 51, 'name': 'bowl'},
    52: {'id': 52, 'name': 'banana'},
    53: {'id': 53, 'name': 'apple'},
    54: {'id': 54, 'name': 'sandwich'},
    55: {'id': 55, 'name': 'orange'},
    56: {'id': 56, 'name': 'broccoli'},
    57: {'id': 57, 'name': 'carrot'},
    58: {'id': 58, 'name': 'hot dog'},
    59: {'id': 59, 'name': 'pizza'},
    60: {'id': 60, 'name': 'donut'},
    61: {'id': 61, 'name': 'cake'},
    62: {'id': 62, 'name': 'chair'},
    63: {'id': 63, 'name': 'couch'},
    64: {'id': 64, 'name': 'potted plant'},
    65: {'id': 65, 'name': 'bed'},
    67: {'id': 67, 'name': 'dining table'},
    70: {'id': 70, 'name': 'toilet'},
    72: {'id': 72, 'name': 'tv'},
    73: {'id': 73, 'name': 'laptop'},
    74: {'id': 74, 'name': 'mouse'},
    75: {'id': 75, 'name': 'remote'},
    76: {'id': 76, 'name': 'keyboard'},
    77: {'id': 77, 'name': 'cell phone'},
    78: {'id': 78, 'name': 'microwave'},
    79: {'id': 79, 'name': 'oven'},
    80: {'id': 80, 'name': 'toaster'},
    81: {'id': 81, 'name': 'sink'},
    82: {'id': 82, 'name': 'refrigerator'},
    84: {'id': 84, 'name': 'book'},
    85: {'id': 85, 'name': 'clock'},
    86: {'id': 86, 'name': 'vase'},
    87: {'id': 87, 'name': 'scissors'},
    88: {'id': 88, 'name': 'teddy bear'},
    89: {'id': 89, 'name': 'hair drier'},
    90: {'id': 90, 'name': 'toothbrush'},
}


def downloadModel(MODEL_URL):
    firstpos = MODEL_URL.rfind("/")
    lastpos = MODEL_URL.rfind(".")
    MODEL_NAME = MODEL_URL[firstpos + 1:lastpos]
    MODEL_FILE = MODEL_NAME + '.tar.gz'

    print("Preparing to download tensorflow model {}".format(MODEL_FILE))
    print("Is file downloaded? %s " % os.path.isfile(MODEL_FILE))
    if not os.path.isfile(MODEL_FILE):
        opener = urllib.request.URLopener()
        opener.retrieve(MODEL_URL, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        tar_file.extract(file, os.getcwd())


def loadTensorflowModel():
    start_time = time.time()
    tf.keras.backend.clear_session()
    detect_fn = tf.saved_model.load(
        '/tensorflow/models/research/object_detection/test_data/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model/')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Elapsed time: ' + str(elapsed_time) + 's')
    return detect_fn

def doinference(image_np):
    input_tensor = np.expand_dims(image_np, 0)
    start_time = time.time()
    detections = detect_fn(input_tensor)
    end_time = time.time()
    elapsed.append(end_time - start_time)

    plt.rcParams['figure.figsize'] = [42, 21]
    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.int32),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.40,
        agnostic_mode=False)

    return image_np_with_detections

MODEL_URL="http://download.tensorflow.org/models/object_detection/tf2/20210210/centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz"

os.chdir("/tensorflow/models/research/object_detection/test_data/")
downloadModel(MODEL_URL)
detect_fn= loadTensorflowModel()
# start of main code
# read from camera

from picamera.array import PiRGBArray
from picamera import PiCamera
# initialize the camera and grab a reference to the raw camera capture

# FOR ARDUNIO
import pyfirmata
from time import sleep
import math
#from tkinter import *

#First engine
global pin6
global pin5
global pin4

#Second engine
global pin9
global pin10
global pin11
        
cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)
stat = cap.isOpened()
print(stat)

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# App Globals (do not edit)
app = Flask(__name__,template_folder='templates')

@app.route('/path')
def path():
    return app.instance_path #you can customze index.html here

@app.route('/')
def index():
    print(app.instance_path)
    return render_template('index.html') #you can customze index.html here

def gen(frame_rate_calc):
    #get camera frame

    while(True):
        
        t1 = cv2.getTickCount()
        ret, frame = c.read()
        #frame = imutils.resize(frame, width=512)
        
        image_rgb_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb_np_with_detections = doinference(image_rgb_np)
        cv2.putText(image_rgb_np_with_detections,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
       
        jpeg = cv2.imencode('.jpg', image_rgb_np_with_detections)[1].tobytes()
        
        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n\r\n')
    
@app.route('/video_feed')
def video_feed():
    return Response(gen(freq),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/left_move')
def left_move():
    pin5.write(0)
    pin4.write(1)
    pin10.write(1)
    pin11.write(0)
    a = request.args.get('a', 0, type=int)
    b = request.args.get('b', 0, type=int)
    print('left')
    return jsonify(result=a + b)
@app.route('/right_move')
def right_move():
    pin5.write(1)
    pin4.write(0)
    pin10.write(0)
    pin11.write(1)
    a = request.args.get('a', 0, type=int)
    b = request.args.get('b', 0, type=int)
    print('right')
    return jsonify(result=a + b)
@app.route('/up_move')
def up_move():
    pin5.write(1)
    pin4.write(0)
    pin10.write(1)
    pin11.write(0)
    sleep(5)
    a = request.args.get('a', 0, type=int)
    b = request.args.get('b', 0, type=int)
    print('up')
    return jsonify(result=a + b)
@app.route('/down_move')
def down_move():
    pin5.write(0)
    pin4.write(0)
    pin10.write(0)
    pin11.write(0)
    pin6.write(0)
    sleep(5)
    print('down')
    return jsonify(result=a + b)

if __name__ == '__main__':

    app.run(host='0.0.0.0', debug=False)
    


