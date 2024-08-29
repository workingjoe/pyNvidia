#!/usr/bin/python3 
# note: this is using BASE python3.6 where pytorch and torchvision are installed -- 

# get details on USB camera attached and see what formats are supported
# v4l2-ctl -d /dev/video2 --list-formats-ext
# ex: 640x480 @ 30fps  YUYV 4:2:2
 
import jetson_inference
import jetson_utils

# use training model we downloaded

# n_net = jetson_inference.detectNet('ssd-mobilenet-v2', threshold=0.5)
n_net = jetson_inference.imageNet('googlenet')

# cam_0 = jetson_utils.gstCamera(0) # CSI camera
cam  = jetson_utils.gstCamera(640, 480, '/dev/video2') # USB camera

disp = jetson_utils.glDisplay()

fnt = jetson_utils.cudaFont()

# stay looping while window is there -- press ESC to quit()
while disp.IsOpen():
    # return each frame with dimensions 
    frame, width, height = cam.CaptureRGBA() 
    # try and find stuff in picture.
    classID, confidence = n_net.Classify(frame, width, height)
    # now lookup classID to string
    item = n_net.GetClassDesc(classID)
    # stuff item text into frame
    fnt.OverlayText(frame, width, height, item, 5, 5, fnt.Magenta, fnt.Black)

    disp.RenderOnce(frame, width, height)


