# on the Raspberry Pi-Zero-W 

# for normal version 2.0 CSI camera
# no timeout, 1280x720 2Mbps auto-white balance no-local-display output to STDOUT
# h264parse piped to rtp (not rtsp) config ?? 
# tcpserver on  -- port 8554
raspivid -t 0 -w 1280 -h 720 -fps 30 -b 2000000 -awb auto -n -o - | gst-launch-1.0 -v fdsrc ! h264parse ! rtph264pay config-interval=1 pt=96 ! gdppay ! tcpserversink host=0.0.0.0 port=8554

# ===========================================================================

# On JetsonNano -- deepLearning12.py 
  
import cv2
print(cv2.__version__)

dispW=1280
dispH=720
flip=0
#Uncomment These next Two Line for Pi Camera
#camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
#cam= cv2.VideoCapture(camSet)

#WEB cam, uncomment the next line
#(If it does not work, try setting to '1' instead of '0')
#cam=cv2.VideoCapture(0)

# get REMOTE camera from Rpi-Z webcam
camRemote =' tcpclientsrc host=192.168.0.15 port=8554 ! gdpdepay ! rtph264depay ! h264parse ! nvv4l2decoder  ! nvvidconv flip-method='+str(flip)+' ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+',format=BGR ! appsink  drop=true sync=false '
cam = camRemote

while True:
    ret, frame = cam.read()
    cv2.imshow('nanoCam',frame)
	cv2.moveWindow('nanoCam', 0, 0)
	
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()  
