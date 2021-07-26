import cv2
import mediapipe as mp
import time
import HolisticModule as hm

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = hm.HolisticDetector()

turtle_neck_count = 0

while True:
    success, img = cap.read()

    img = detector.findHolistic(img, draw=True)

    pose_lmList = detector.findPoseLandmark(img, draw=True)
    face_lmList = detector.findFaceLandmark(img, draw=True)

    if len(pose_lmList) != 0 and len(face_lmList) != 0:
        # print("pose[11]", pose_lmList[11])
        # print("pose[12]", pose_lmList[12])
        # print("face[152]",face_lmList[152])

        center_shoulder = detector.findCenter(11,12)
        length, img = detector.findDistance(152, center_shoulder, img, draw=True)
        # print(length)
        if length < 100:
            turtle_neck_count += 1

        if length < 100 and turtle_neck_count > 100:
            print("WARNING - Keep your posture straight.")
            print("TurtleNeck Point = ", int(length))
            turtle_neck_count = 0

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break 

cap.release()
cv2.destroyAllWindows()
    