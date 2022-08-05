import cv2
import sys
from cv2 import VideoCapture

cascPath = sys.argv[0] #list in python that contains command-line arguments passed to script
                        #filename/script executed 
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


Video_capture = cv2.VideoCapture(0) #sets video source to default webcam

while True:
    #Capture frame-by-frame
    ret, frame = Video_capture.read() #reads one frame from video source 

    vid = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #conver an image from one color space to another

    detect_faces = faceCascade.detectMultiScale(
        vid,
        scaleFactor=1.1, #how much image size is reduced at each image scale
        minNeighbors=5, #specify how many nieghbors each candidate rectangle should have to retain it 
                        #higher value = fewer detections but with higher quality
                        
        minSize=(30, 30), #determines how small size you want to detect
        flags=cv2.CASCADE_SCALE_IMAGE
    )

     # Draw a rectangle around the faces
    for (x, y, w, h) in detect_faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): #returning the number representing the unicode of a specified character
        break

# When everything is done, release the capture
Video_capture.release()
cv2.destroyAllWindows()
