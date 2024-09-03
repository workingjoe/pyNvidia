#!/usr/bin/python3 
# note: this is using BASE python3.6 where pytorch and torchvision are installed -- 

# get details on USB camera attached and see what formats are supported
# v4l2-ctl -d /dev/video2 --list-formats-ext
# ex: 640x480 @ 30fps  YUYV 4:2:2
#
# look at Hello AI world -- in gethub.com/dusty-nv/jetson-inference
#

# NOTE: this version uses OpenCV for gstreamer Camera, textOverlay and display
# Therefore each Frame must be converted via numpy for use by classifier 
# then OpenCV can display the original Frame, and we add Text from classifier.
#
import cv2
import numpy as np 

import jetson_inference
import jetson_utils
import time

from rich import print, pretty, inspect # console, table
# note: now have locals(), inspect(myVar, methods=True) and much more


width  = 640
height = 480
dispW = width
dispH = height

# create a gstreamer pipeline command for CSI cameras
def gstreamer_pipeline(
    sensor_id: int=0,
    capture_width: int=1920,
    capture_height: int=1080,
    display_width: int=1920,
    display_height: int=1080,
    framerate: int=30,
    flip_method: int=0,
) -> str:
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=%d, height=%d, framerate=%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=%d, height=%d, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height
        )
    )

flip  = 0

# horrible string ... one long gstreamer launch
# camSet0='nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method=' + str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
# camSet1='nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method=' + str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
#
# for arducam there is a lens/camera calibration to setup...
# https://docs.arducam.com/Nvidia-Jetson-Camera/Application-note/Fix-Red-Tint-with-ISP-Tuning/
#
# wget https://www.arducam.com/downloads/Jetson/Camera_overrides.tar.gz
# tar zxvf Camera_overrides.tar.gz
# sudo cp camera_overrides.isp /var/nvidia/nvcam/settings/
# sudo chmod 664 /var/nvidia/nvcam/settings/camera_overrides.isp
# sudo chown root:root /var/nvidia/nvcam/settings/camera_overrides.isp


# camSet0 = gstreamer_pipeline( sensor_id=0, 
#                              capture_width=3264,
#                              capture_height=2464,
#                              display_width=dispW,
#                              display_height=dispH,
#                              framerate=21,
#                              flip_method=flip,
#                            )
camSet0 = gstreamer_pipeline( sensor_id=0, 
                             capture_width=1920,
                             capture_height=1080,
                             display_width=dispW,
                             display_height=dispH,
                             framerate=30,
                             flip_method=flip,
                           )
camSet1 = gstreamer_pipeline( sensor_id=1, 
                             capture_width=1920,
                             capture_height=1080,
                             display_width=dispW,
                             display_height=dispH,
                             framerate=30,
                             flip_method=flip,
                           )

# cam = cv2.VideoCapture(camSet1)

# or create USB webcam object (arg is 0,1, or 2 depends on CSI cameras installed)
# check v4l2-ctl -d2 --list-formats-ext and see what parms work...
# camSet2 = 'gst-launch-1.0 v4l2src device=/dev/video2 ! image/jpeg,format=MJPG,width=640,height=480,framerate=30/1 ! nvv4l2decoder mjpeg=1 ! nvvidconv ! autovideosink'
camSet2 = '/dev/video2' # 2

# now assign default camera
camSet  = camSet1 # camSet0, camSet1, camSet2
cam = cv2.VideoCapture(camSet)

# fixup USB camera to reasonable size and formatting
if 'cam' in locals() or 'cam' in globals():
    if (camSet == camSet2) and (cam.isOpened()) :
        width  = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    #     print(f"Width: {width} x Height: {height}")
        if (width != dispW):
            ret = cam.set(cv2.CAP_PROP_FRAME_WIDTH,  dispW)
            # if not ret:
            #     print("can't set WIDTH")
        if (height != dispH):
            ret = cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)
            # if not ret:
            #     print("can't set HEIGHT")

        ret = cam.set(cv2.CAP_PROP_CONVERT_RGB, True)
        # if not ret:
        #     print("can't set CONVERT_RGB")

        width  = dispW # assume cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = dispH # assume cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # print(f"New Width: {width} x New Height: {height}")



# cam = jetson_utils.gstCamera(width, height, '/dev/video2') # USB camera
#
# cam = jetson_utils.gstCamera(width, height, '0') # CSI camera
# cam = jetson_utils.gstCamera(width, height, '1') # CSI camera

# n_net = jetson_inference.detectNet('ssd-mobilenet-v2', threshold=0.5)
n_net = jetson_inference.imageNet('googlenet')
# n_net = jetson_inference.imageNet('alexnet')

# disp = jetson_utils.glDisplay()
# fnt = jetson_utils.cudaFont()
fnt = cv2.FONT_HERSHEY_SIMPLEX


timeMark = time.time()
fpsFilter = 0

while True:
    # return each frame with dimensions (note: not the BGR used in OpenCV)
    # frame, width, height = cam.CaptureRGBA(zeroCopy=1) # special option  
    ret, frame = cam.read()
    imgRGBA = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA).astype(np.float32)
    img = jetson_utils.cudaFromNumpy(imgRGBA)

    # try and find stuff in picture.
    classID, confidence = n_net.Classify(img, width, height)

    # now lookup classID to string
    # if (confidence)
    item = n_net.GetClassDesc(classID)
    
    deltaT = time.time() - timeMark
    fps = 1/deltaT
    fpsFilter = 0.95*fpsFilter + 0.05*fps
    timeMark = time.time()

    # stuff item text into frame
    # fnt.OverlayText(frame, width, height, str(round(fpsFilter,1))+' fps ' + item, 5, 5, fnt.Yellow, fnt.Black)
    # fnt.OverlayText(frame, width, height, item, 5, 5, fnt.Magenta, fnt.Black)

    # disp.RenderOnce(frame, width, height)

    # convert from CUDA format to OpenCV BGR
    # frame = jetson_utils.cudaToNumpy(frame, width, height, 4) # R-G-B-A is 4 elements
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR).astype(np.uint8) # from float to uint8 ??
    cv2.putText(frame, str(round(fpsFilter,1))+' fps ' + item,(0,30),fnt, 1, (0,255,255),2)
    cv2.imshow('RecognizeCam', frame )
    cv2.moveWindow('RecognizeCam',0,0)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
