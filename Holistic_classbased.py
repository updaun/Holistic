import cv2
import mediapipe as mp
import time
import HolisticModule as hm

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = hm.HolisticDetector()

while True:
    success, img = cap.read()

    img = detector.findHolistic(img, draw=False)

   
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break 

cap.release()
cv2.destroyAllWindows()
    