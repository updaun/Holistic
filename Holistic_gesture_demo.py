import cv2
from win10toast import ToastNotifier
import os
import math
import modules.HolisticModule as hm
from modules.turtle_neck import turtlenect_detection
from modules.sleep_detect_angle import sleepiness_detection
from modules.fps import fps_present


# video input 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

folderPath = "expression_image"
myList = os.listdir(folderPath)
overlayList = []

o_count = 0
x_count = 0
like_count = 0 
o_present_count = 0
x_present_count = 0
like_present_count = 0

epsilon = 1.0e-10

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
    pose_lmList = detector.findPoseLandmark(img, draw=False)
    # 468개의 얼굴 점 리스트
    face_lmList = detector.findFaceLandmark(img, draw=False)
    
    left_hand_lmList = detector.findLefthandLandmark(img, draw=False)
    right_hand_lmList = detector.findRighthandLandmark(img, draw=False)

    if len(face_lmList) != 0:
        sleepiness_detection(detector, img, log=False, notification=False)

    if len(pose_lmList) != 0 and len(face_lmList) != 0:
        turtlenect_detection(detector, img, sensitivity = 9, log=False, notification=False)

    # 인체가 감지가 되었는지 확인하는 구문
    if len(left_hand_lmList) != 0 and len(right_hand_lmList) != 0:

        # To detect O gesture
        thumb_length = detector.findLength_lh_rh(4, 4)
        index_length = detector.findLength_lh_rh(8, 8)
        left_hand_length = detector.findLength_lh_lh(8,4)
        left_threshold_length = detector.findLength_lh_lh(0,17)
        right_hand_length = detector.findLength_rh_rh(8,4)
        right_threshold_length = detector.findLength_rh_rh(0,17)

        # To detect X gesture
        left_hand_fingersUp_list_a0 = detector.left_hand_fingersUp(axis=False)
        right_hand_fingersUp_list_a0 = detector.right_hand_fingersUp(axis=False)
        index_mcp_length = detector.findLength_lh_rh(5, 5)
        left_index_length = detector.findLength_lh_rh(7, 7)
        # print(left_hand_fingersUp_list_a0, right_hand_fingersUp_list_a0)
    
        # To detect LIKE gesture
        left_hand_fingersUp_list_a1 = detector.left_hand_fingersUp(axis=True)
        right_hand_fingersUp_list_a1 = detector.right_hand_fingersUp(axis=True)    
        # print(left_hand_fingersUp_list_a1, right_hand_fingersUp_list_a1)

        if len(pose_lmList) != 0:
            wrist_length = detector.findLength_pose(16,15)
            shoulder_length = detector.findLength_pose(12,11)
            cross_length_1 = detector.findLength_pose(15,14)
            cross_length_2 = detector.findLength_pose(16,13)
        

        # O detect
        if left_hand_fingersUp_list_a0[2:] == [1,1,1] and left_hand_fingersUp_list_a0[2:] == [1,1,1] and thumb_length < 50 and index_length < 50 and left_hand_length > left_threshold_length and right_hand_length > right_threshold_length:
            o_count += 1
            x_present_count = 0
            like_present_count = 0

        # O detect
        elif pose_lmList[16][1] < pose_lmList[15][1] and wrist_length < shoulder_length and pose_lmList[16][2] < pose_lmList[10][2] and pose_lmList[15][2] < pose_lmList[9][2]:
            o_count += 1
            x_present_count = 0
            like_present_count = 0
                    
        # X detect
        elif left_hand_fingersUp_list_a0[2:] == [0,0,0] and left_hand_fingersUp_list_a0[2:] == [0,0,0] and right_hand_lmList[8][1]+ 15 > left_hand_lmList[8][1]  and left_index_length < left_threshold_length and index_mcp_length < 150:
            x_count += 1
            o_present_count = 0
            like_present_count = 0
        
        # X detect
        elif pose_lmList[11][1] > pose_lmList[12][1] and pose_lmList[16][1] > pose_lmList[15][1] and cross_length_1 + cross_length_2 > wrist_length*2:
            x_count += 1
            o_present_count = 0
            like_present_count = 0
        
        # LIKE detect
        elif left_hand_fingersUp_list_a1 == [1,0,0,0,0] and left_hand_fingersUp_list_a1 == [1,0,0,0,0] and right_hand_lmList[4][2] < right_hand_lmList[2][2] and left_hand_lmList[4][2] < left_hand_lmList[2][2]:
            like_count += 1
            o_present_count = 0
            x_present_count = 0
    
    else:
        if len(pose_lmList) != 0:
            wrist_length = detector.findLength_pose(16,15)
            shoulder_length = detector.findLength_pose(12,11)
            cross_length_1 = detector.findLength_pose(15,14)
            cross_length_2 = detector.findLength_pose(16,13)

            if pose_lmList[16][1] < pose_lmList[15][1] and wrist_length < shoulder_length and pose_lmList[16][2] < pose_lmList[10][2] and pose_lmList[15][2] < pose_lmList[9][2]:
                o_count += 1
                x_present_count = 0
                like_present_count = 0

            # X detect
            elif pose_lmList[11][1] > pose_lmList[12][1] and pose_lmList[16][1] > pose_lmList[15][1] and cross_length_1 + cross_length_2 > wrist_length*2:
                x_count += 1
                o_present_count = 0
                like_present_count = 0

    # count control
    if o_count > 10:
        o_present_count = 15
        o_count = 0
        x_count = 0
        like_count = 0
    
    if x_count > 10:
        x_present_count = 15
        o_count = 0
        x_count = 0
        like_count = 0

    if like_count > 10:
        like_present_count = 15
        o_count = 0
        x_count = 0
        like_count = 0


    if o_present_count > 0:
        h, w, c = overlayList[0].shape
        img[15:h+15, 15:w+15] = overlayList[0]
        cv2.rectangle(img, (0, 0), (int(cap.get(3)), int(cap.get(4))), (225, 125, 75), 30)
        o_present_count -= 1

    if x_present_count > 0:
        h, w, c = overlayList[1].shape
        img[15:h+15, 15:w+15] = overlayList[1]
        cv2.rectangle(img, (0, 0), (int(cap.get(3)), int(cap.get(4))), (55, 55, 240), 30)
        x_present_count -= 1

    if like_present_count > 0:
        h, w, c = overlayList[2].shape
        img[15:h+15, 15:w+15] = overlayList[2]
        cv2.rectangle(img, (0, 0), (int(cap.get(3)), int(cap.get(4))), (40, 220, 30), 30)
        like_present_count -= 1
        
        # turtlenect_detection(detector, img, sensitivity = 8, log=False, notification=True)

        # eyeblink_detection(detector, img, sensitivity = 10, log=True, notification=True)

    # fps_present(img, draw=True)                                                                                                                                                                          


    img = cv2.flip(img, 1)

    # img를 우리에게 보여주는 부분
    cv2.imshow("Image", img)

    # ESC 키를 눌렀을 때 창을 모두 종료하는 부분
    if cv2.waitKey(1) & 0xFF == 27:
        break 

cap.release()
cv2.destroyAllWindows()
    