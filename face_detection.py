import numpy
import cv2 as cv

face = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv.imread('bppt2.jpg')
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

detection_face = face.detectMultiScale(img_gray,1.1,5)
font = cv.FONT_HERSHEY_SIMPLEX
total = 0

for(x,y,w,h) in detection_face:
    total += 1
    cv.putText(img,"Face",(x,y-10),font,0.75,(0,0,255),2,cv.LINE_AA)
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = img_gray[y:y+h,x:x+w]
    roi_color = img[y:y+h,x:x+w]

cv.putText(img,"Total face is : "+str(total)+ " face",(10,30),font,1,(0,0,0),2,cv.LINE_AA)
cv.imShow('img',img)
cv.waitKey(1)
cv.destroyAllWindows()
