
#FOR WEB TRANSFER
from flask import Flask, jsonify,render_template, Response, request
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import threading
import os

#FOR CAMERA VIEW
from picamera.array import PiRGBArray
from picamera import PiCamera

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

#declare board
board=pyfirmata.Arduino('/dev/ttyACM0')

pin6 = board.get_pin('d:6:i')
pin5 = board.digital[5]
pin4 = board.digital[4]

pin9 = board.get_pin('d:9:i')
pin10 = board.digital[10]
pin11 = board.digital[11]

# initialize the camera and grab a reference to the raw camera capture
flagEnd = False;
        
cap = cv2.VideoCapture(0)

stat = cap.isOpened()
print(stat)
# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

        
# App Globals (do not edit)
app = Flask(__name__)

@app.route('/')
def index():
    print(app.instance_path)
    return render_template('index.html') #you can customze index.html here

def gen(frame_rate_calc):
    #get camera frame

    while(True):
        
        t1 = cv2.getTickCount()
        ret, frame = cap.read()
        #image_rgb_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
       
        jpeg = cv2.imencode('.jpg', frame)[1].tobytes()
        
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
    


