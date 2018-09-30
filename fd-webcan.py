# https://towardsdatascience.com/face-detection-for-beginners-e58e8f21aad9https://towardsdatascience.com/face-detection-for-beginners-e58e8f21aad9
# import libraries
import cv2 as cv2
import numpy as np

#import classifier for face and eye detection
face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
# Import Classifier for Face and Eye Detection
# face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier ('Haarcascades/haarcascade_eye.xml')

def face_detector (img, size=0.5,count=0):
    img=cv2.flip(img,1)
    # Convert Image to Grayscale
    print ('*** convert image into gray***',count)
    # print(img)
    # if(img==None) :
    #    return img
    gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
    print ('*** convert image into gray ... done')

    # faces = face_classifier.detectMultiScale (gray, 1.3, 5)
    faces = face_classifier.detectMultiScale (gray, 1.2, 2)
    if faces is ():
        return img
    
    # Given coordinates to detect face and eyes location from ROI
    for (x, y, w, h) in faces:
        #x = x - 100
        #w = w + 100
        #y = y - 100
        #h = h + 100
        cv2.rectangle (img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y: y+h, x: x+w]
        roi_color = img[y: y+h, x: x+w]
        eyes = eye_classifier.detectMultiScale (roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
    # roi_color = cv2.flip (roi_color, 1)
    # return roi_color
    return img
# Webcam setup for Face Detection
cap = cv2.VideoCapture (0)
count=0
while True:
    count=count+1
    ret, frame = cap.read ()
    faces=face_detector (frame,count=count)
    # if ( faces == None ) : 
    #    continue
    cv2.imshow ('Our Face Extractor', faces)
    key=cv2.waitKey(100)
    if key==13 or key==10 or key==32 : #13 is the Enter Key
        break

    # When everything done, release the capture
cap.release ()
cv2.destroyAllWindows ()


