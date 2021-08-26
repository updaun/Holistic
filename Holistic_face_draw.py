import cv2
from win10toast import ToastNotifier

import modules.HolisticModule as hm
from modules.fps import fps_present

import numpy as np


# video input 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

canvas = np.zeros((int(cap.get(4)), int(cap.get(3)), 3), np.uint8)

# Holistic 객체(어떠한 행위를 하는 친구) 생성
detector = hm.HolisticDetector()

img_counter = 1

while True:
    # defalut BGR img
    success, img = cap.read()
    # mediapipe를 거친 이미지 생성 -> img
    img = detector.findHolistic(img, draw=False)
    # output -> list ( id, x, y, z) 32 개 좌표인데 예를 들면, (11, x, y, z)
    pose_lmList = detector.findPoseLandmark(img, draw=False)
    # 468개의 얼굴 점 리스트
    face_lmList = detector.findFaceLandmark(img, draw=False)
    
    draw_list = [
                    [98, 97, 2 , 326, 327], # nose
                    [168, 4], # nose center
                    [46, 53, 52, 65, 55], # left eyebrow
                    [285, 295, 282, 283, 276], # right eyebrow
                    [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33], # left eye
                    [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362], # right eye
                    [13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 183, 191, 80, 81, 82, 13], # mouth inner
                    [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37, 0], # mouth outer
                    [389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 227, 34, 139], # face line
                ]

    # 인체가 감지가 되었는지 확인하는 구문
    if len(pose_lmList) != 0 and len(face_lmList) != 0:
        for part in draw_list:
            for i in range(len(part)-1):
                detector.drawLine(part[i], part[i+1], img, t=2)
                # detector.drawLine(part[i], part[i+1], canvas, t=2)

    fps_present(img, draw=False)

    # img를 우리에게 보여주는 부분
    cv2.imshow("Image", img)

    # ESC 키를 눌렀을 때 창을 모두 종료하는 부분
    if cv2.waitKey(1) & 0xFF == 27:
        if len(pose_lmList) != 0 and len(face_lmList) != 0:
            for part in draw_list:
                for i in range(len(part)-1):
                    detector.drawLine(part[i], part[i+1], canvas, t=2)
        cv2.imwrite(f'./output_image/canvas_{img_counter}.png', canvas)
        print("Save Canvas Successfully")
        break 

    # Spacebar 또는 Return 누르면 whiteCanvas 저장
    if cv2.waitKey(1) & 0xFF == 32 or cv2.waitKey(1) & 0xFF == 13:
        if len(pose_lmList) != 0 and len(face_lmList) != 0:
            for part in draw_list:
                for i in range(len(part)-1):
                    detector.drawLine(part[i], part[i+1], canvas, t=2)
        cv2.imwrite(f'./output_image/canvas_{img_counter}.png', canvas)
        print("Save Canvas Successfully")
        img_counter += 1

cap.release()
cv2.destroyAllWindows()
    