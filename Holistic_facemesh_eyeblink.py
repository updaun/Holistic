import cv2
import mediapipe as mp
import time
import HolisticModule as hm
from win10toast import ToastNotifier
import math
import numpy as np

###################################################
sensitivity = 8
###################################################

# privious time for fps
pTime = 0
# cerrent time for fps
cTime = 0

# video input 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Holistic 객체(어떠한 행위를 하는 친구) 생성
detector = hm.HolisticDetector()

# toast 알림을 주는 객체 생성
toaster = ToastNotifier()

# turtle_neck_count 변수 초기 세팅
turtle_neck_count = 0

eye_blink_count = 0

while True:
    # defalut BGR img
    success, img = cap.read()

    # mediapipe를 거친 이미지 생성 -> img
    img = detector.findHolistic(img, draw=False)

    # output -> list ( id, x, y, z) 32 개 좌표인데 예를 들면, (11, x, y, z)
    pose_lmList = detector.findPoseLandmark(img, draw=False)
    # 468개의 얼굴 점 리스트
    face_lmList = detector.findFaceLandmark(img, draw=False)
    
    
    
    # 인체가 감지가 되었는지 확인하는 구문
    if len(pose_lmList) != 0 and len(face_lmList) != 0:
        right_eye_length, img = detector.findEyeBlink(159, 145, img, draw=True, r=5, t=2)
        left_eye_length, img = detector.findEyeBlink(386, 374, img, draw=True, r=5, t=2)
        
        eye_depth = detector.findEyeDepth(8, 9)
        
        eye_blink_threshold = eye_depth 
        # eye_blink_threshold = np.interp(eye_blink_threshold, [0, 15], [0, 80])

        if right_eye_length*2 < eye_blink_threshold and left_eye_length*2 < eye_blink_threshold:
            eye_blink_count += 1
        print("length : ", round(right_eye_length*2), "  " ,round(left_eye_length*2), "eye_blink_threshold :", round(eye_blink_threshold, 2), "eye_blink_count :", eye_blink_count)
        # print("right_eye_length : ", round(right_eye_length**2), "left_eye_length : ", round(left_eye_length**2), "eye_blink_threshold :", round(eye_blink_threshold, 2), "eye_blink_count :", eye_blink_count)

    # fps 계산 로직
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    # fps를 이미지 상단에 입력하는 로직
    cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    # img를 우리에게 보여주는 부분
    cv2.imshow("Image", img)

    # ESC 키를 눌렀을 때 창을 모두 종료하는 부분
    if cv2.waitKey(1) & 0xFF == 27:
        break 

cap.release()
cv2.destroyAllWindows()
    