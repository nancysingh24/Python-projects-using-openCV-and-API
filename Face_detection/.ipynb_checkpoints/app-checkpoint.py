import cv2 as cv
import numpy as np
#taking camera resource from the system
cam=cv.VideoCapture(0)
#detector
detector=cv.CascadeClassifier ("haar_frontal.xml")
def main():
    while True:
        #reading the frames
        _,frame=cam.read()
        #downscale if u need it
        #frame=cv.resize(frame,None,0.8,0.8,cv.INTER_LINEAR)
        gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        # detect the faces
        faces=detector.detectMultiScale(gray,1.3,5)
        for face in faces:
            x,y,w,h=face
            cv.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
        #faces=detector.detect.detectMUltiscale(gray,)
        #showing the frame

        cv.imshow("face detector",frame)
        key=cv.waitKey(5)
        if key & 0xff == ord("q"):
            cv.destroyAllWindows()
            break
    cam.release()

if __name__=="__main__":
    main()
