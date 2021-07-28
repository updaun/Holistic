import cv2
import mediapipe as mp
import time
import HolisticModule as hm
from win10toast import ToastNotifier
import math

###################################################
sensitivity = 6
###################################################

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = hm.HolisticDetector()

toaster = ToastNotifier()

turtle_neck_count = 0


while True:
    success, img = cap.read()

    img = detector.findHolistic(img, draw=False)

    pose_lmList = detector.findPoseLandmark(img, draw=False)
    face_lmList = detector.findFaceLandmark(img, draw=False)
    

    if len(pose_lmList) != 0 and len(face_lmList) != 0:
        # print("pose[11]", pose_lmList[11])
        # print("pose[12]", pose_lmList[12])
        # print("face[152]",face_lmList[152])

        center_shoulder = detector.findCenter(11,12)
        length, img = detector.findDistance(152, center_shoulder, img, draw=False)
        pose_depth = 500 - detector.findDepth(11,12)

        # if pose_depth < 200:
        #     turtleneck_detect_threshold = 55
        # else:
        #     turtleneck_detect_threshold = 70

        # turtleneck_detect_threshold = pose_depth / 4
        turtleneck_detect_threshold = abs(math.log2(pose_depth)) * sensitivity
        
        # print("Length : {:.3f},   Threshold : {:.3f},   Pose_depth : {}".format(length, turtleneck_detect_threshold, pose_depth))
    


        if length < turtleneck_detect_threshold:
            turtle_neck_count += 1

        if length < turtleneck_detect_threshold and turtle_neck_count > 100:
            tutleneck_score = int((turtleneck_detect_threshold - int(length))/turtleneck_detect_threshold*100)
            print("WARNING - Keep your posture straight.")
            print("TurtleNeck Score = ", tutleneck_score)
            toaster.show_toast("TurtleNect WARNING", f"Keep your posture straight.\n\nDegree Of TurtleNeck = {tutleneck_score}")
            turtle_neck_count = 0

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    # cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break 

cap.release()
cv2.destroyAllWindows()
    