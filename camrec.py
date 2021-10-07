'''
cap=cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

while(True):
	ret,frame=cap.read()
	cv2.imshow('frame',frame)
	if ret:
		frame = cv2.resize(frame, (640, 480))
		out.write(frame)
		cv2.imshow('Video', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break

cap.release()
out.release()
cv2.destroyAllWindows()

'''
import cv2
import sys
import numpy as np
import time
import imutils

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
bodyCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
video_capture = cv2.VideoCapture(0)
k=0

while True:
    # Capture frame-by-frame
    ret,frame = video_capture.read()
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    bodies=bodyCascade.detectMultiScale(gray,1.3,5)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
    	cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    	fname=str("test")+str(k)+str(".mp4")
    	out = cv2.VideoWriter(fname,fourcc, 20.0, (640, 480))
    	#recos(k)
    	k+=1
    	capture_duration = 5 #records for a minimum of 5 seconds when a face detected
    	start_time = time.time()
    	while( int(time.time() - start_time) < capture_duration ):
    		ret,frame = video_capture.read()
    		frame = cv2.resize(frame, (640, 480))
    		if ret:
    			out.write(frame)
    	out.release()
    	
    '''
    for (x, y, w, h) in bodies:
    	cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    	fname=str("test")+str(k)+str(".mp4")
    	out = cv2.VideoWriter(fname,fourcc, 20.0, (640, 480))
    	#recos(k)
    	k+=1
    	capture_duration = 5 #records for a minimum of 5 seconds when a face detected
    	start_time = time.time()
    	while( int(time.time() - start_time) < capture_duration ):
    		ret,frame = video_capture.read()
    		frame = cv2.resize(frame, (640, 480))
    		if ret:
    			out.write(frame)
    	out.release()
    '''

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
out.release()
cv2.destroyAllWindows()