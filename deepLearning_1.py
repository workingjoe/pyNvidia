#!/usr/bin/python3 
# note: this is using BASE python3.6 where pytorch and torchvision are installed -- 

# get details on USB camera attached and see what formats are supported
# v4l2-ctl -d /dev/video2 --list-formats-ext
# ex: 640x480 @ 30fps  YUYV 4:2:2
#
# look at Hello AI world -- in gethub.com/dusty-nv/jetson-inference
#
 
import jetson_inference
import jetson_utils
import time
from rich import print, pretty, inspect # console, table
# note: now have locals(), inspect(myVar, methods=True) and much more


width = 1280
height = 720
# cam = jetson_utils.gstCamera(width, height, '/dev/video2') # USB camera
#
# cam = jetson_utils.gstCamera(width, height, '0') # CSI camera
cam = jetson_utils.gstCamera(width, height, '1') # CSI camera

# n_net = jetson_inference.detectNet('ssd-mobilenet-v2', threshold=0.5)
n_net = jetson_inference.imageNet('googlenet')

disp = jetson_utils.glDisplay()

fnt = jetson_utils.cudaFont()

timeMark = time.time()
fpsFilter = 0

while disp.IsOpen():
    # return each frame with dimensions (note: not the BGR used in OpenCV)
    frame, width, height = cam.CaptureRGBA() 
    # try and find stuff in picture.
    classID, confidence = n_net.Classify(frame, width, height)

    # now lookup classID to string
    # if (confidence)
    item = n_net.GetClassDesc(classID)
    
    deltaT = time.time() - timeMark
    fps = 1/deltaT
    fpsFilter = 0.95*fpsFilter + 0.05*fps
    timeMark = time.time()

    # stuff item text into frame
    fnt.OverlayText(frame, width, height, str(round(fpsFilter,1))+' fps ' + item, 5, 5, fnt.Yellow, fnt.Black)
    # fnt.OverlayText(frame, width, height, item, 5, 5, fnt.Magenta, fnt.Black)

    disp.RenderOnce(frame, width, height)
