import cv2
from win10toast import ToastNotifier
import os
import math
import modules.HolisticModule as hm
from modules.turtle_neck import turtlenect_detection
from modules.eye_blink import eyeblink_detection
from modules.fps import fps_present


# video input 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

folderPath = "expression_image"
myList = os.listdir(folderPath)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    print(f'{folderPath}/{imPath}')
    overlayList.append(image)

# Holistic 객체(어떠한 행위를 하는 친구) 생성
detector = hm.HolisticDetector()

while True:
    # defalut BGR img
    success, img = cap.read()
    # mediapipe를 거친 이미지 생성 -> img
    img = detector.findHolistic(img, draw=True)
    # output -> list ( id, x, y, z) 32 개 좌표인데 예를 들면, (11, x, y, z)
    # pose_lmList = detector.findPoseLandmark(img, draw=False)
    # 468개의 얼굴 점 리스트
    # face_lmList = detector.findFaceLandmark(img, draw=False)
    
    left_hand_lmList = detector.findLefthandLandmark(img, draw=False)
    right_hand_lmList = detector.findRighthandLandmark(img, draw=False)

    

    # 인체가 감지가 되었는지 확인하는 구문
    if len(left_hand_lmList) != 0 and len(right_hand_lmList) != 0:
        # print(left_hand_lmList)
        # print(right_hand_lmList)
        thumb_length = math.hypot(abs(right_hand_lmList[4][1]-left_hand_lmList[4][1]), abs(right_hand_lmList[4][2]-left_hand_lmList[4][2]))
        index_length = math.hypot(abs(right_hand_lmList[8][1]-left_hand_lmList[8][1]), abs(right_hand_lmList[8][2]-left_hand_lmList[8][2]))
        index_pip_length = math.hypot(abs(right_hand_lmList[6][1]-left_hand_lmList[6][1]), abs(right_hand_lmList[6][2]-left_hand_lmList[6][2]))

        index_mcp_length = math.hypot(abs(right_hand_lmList[5][1]-left_hand_lmList[5][1]), abs(right_hand_lmList[5][2]-left_hand_lmList[5][2]))
        print(index_mcp_length)

        left_threshold_length = math.hypot(abs(left_hand_lmList[0][1]-left_hand_lmList[17][1]), abs(left_hand_lmList[0][2]-left_hand_lmList[17][2]))
        left_hand_length = math.hypot(abs(left_hand_lmList[8][1]-left_hand_lmList[4][1]), abs(left_hand_lmList[8][2]-left_hand_lmList[4][2]))
        
        right_threshold_length = math.hypot(abs(right_hand_lmList[0][1]-right_hand_lmList[17][1]), abs(right_hand_lmList[0][2]-right_hand_lmList[17][2]))
        right_hand_length = math.hypot(abs(right_hand_lmList[8][1]-right_hand_lmList[4][1]), abs(right_hand_lmList[8][2]-right_hand_lmList[4][2]))



        if thumb_length < 50 and index_length < 50 and left_hand_length > left_threshold_length and right_hand_length > right_threshold_length and index_pip_length > 50:
            h, w, c = overlayList[0].shape
            img[15:h+15, 15:w+15] = overlayList[0]
            cv2.rectangle(img, (0, 0), (int(cap.get(3)), int(cap.get(4))), (225, 125, 75), 30)
        # print(index_length)

        if right_hand_lmList[8][1] > left_hand_lmList[8][1] + 30 and index_mcp_length < 150:
            h, w, c = overlayList[1].shape
            img[15:h+15, 15:w+15] = overlayList[1]
            cv2.rectangle(img, (0, 0), (int(cap.get(3)), int(cap.get(4))), (55, 55, 240), 30)

        pass
        # turtlenect_detection(detector, img, sensitivity = 8, log=False, notification=True)

        # eyeblink_detection(detector, img, sensitivity = 10, log=True, notification=True)

    # fps_present(img, draw=True)

    # img를 우리에게 보여주는 부분
    cv2.imshow("Image", img)

    # ESC 키를 눌렀을 때 창을 모두 종료하는 부분
    if cv2.waitKey(1) & 0xFF == 27:
        break 

cap.release()
cv2.destroyAllWindows()
    