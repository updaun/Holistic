import cv2
import modules.HolisticModule as hm
from win10toast import ToastNotifier
import math
import time

import dlib
from keras.models import load_model
from imutils import face_utils
import numpy as np


###################################################
sensitivity = 8
###################################################

# privious time for fps
pTime = 0
# cerrent time for fps
cTime = 0

IMG_SIZE = (34, 26)

eyedetector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

model = load_model('models/2018_12_17_22_58_35.h5')

# video input 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Holistic 객체(어떠한 행위를 하는 친구) 생성
detector = hm.HolisticDetector()

# toast 알림을 주는 객체 생성
toaster = ToastNotifier()

# turtle_neck_count 변수 초기 세팅
turtle_neck_count = 0

def crop_eye(img, eye_points):
  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

  eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect

while True:
    # defalut BGR img
    success, img = cap.read()

    img_eye = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(img_eye, cv2.COLOR_BGR2GRAY)
    faces = eyedetector(gray)
    # mediapipe를 거친 이미지 생성 -> img
    img = detector.findHolistic(img, draw=False)

    # output -> list ( id, x, y, z) 32 개 좌표인데 예를 들면, (11, x, y, z)
    pose_lmList = detector.findPoseLandmark(img, draw=True)
    # 468개의 얼굴 점 리스트
    face_lmList = detector.findFaceLandmark(img, draw=True)
    

    # 인체가 감지가 되었는지 확인하는 구문
    if len(pose_lmList) != 0 and len(face_lmList) != 0:
        # print("pose[11]", pose_lmList[11])
        # print("pose[12]", pose_lmList[12])
        # print("face[152]",face_lmList[152])

        # 양 어깨 좌표 11번과 12번의 중심 좌표를 찾아 낸다.
        center_shoulder = detector.findCenter(11,12)

        # 목 길이 center_shoulder 좌표와 얼굴 152번(턱) 좌표를 사용하여 길이 구하는 부분
        # 목 길이가 표시된 이미지로 변경
        length, img = detector.findDistance(152, center_shoulder, img, draw=True)

        # x, y, z좌표 예측 (노트북 웹캠과의 거리를 대강 예측) - 노트북과의 거리
        pose_depth = abs(500 - detector.findDepth(11,12)) 
        # if pose_depth < 200:
        #     turtleneck_detect_threshold = 55
        # else:
        #     turtleneck_detect_threshold = 70

        # turtleneck_detect_threshold = pose_depth / 4
        # 노트북과의 거리는 0보다 커야한다.
        if pose_depth > 0:
            # 거북목 감지 임계치
            turtleneck_detect_threshold = abs(math.log2(pose_depth)) * sensitivity
        # 노트북과의 거리가 아주 가까운 상태
        else:
            turtleneck_detect_threshold = 50
        # 목길이, 임계치, 노트북과의 거리
        print("Length : {:.3f},   Threshold : {:.3f},   Pose_depth : {}".format(length, turtleneck_detect_threshold, pose_depth))
    

        # 핵심 로직 목 길이가 임계치보다 작을 때, 거북목으로 생각한다.
        if length < turtleneck_detect_threshold:
            turtle_neck_count += 1

        # 100번 거북목으로 인식되면 알림을 제공한다. 
        if length < turtleneck_detect_threshold and turtle_neck_count > 100:
            # 얼마나 거북목인지 계산해주는 부분 (0~ 100 점) 
            tutleneck_score = int((turtleneck_detect_threshold - int(length))/turtleneck_detect_threshold*100)
            print("WARNING - Keep your posture straight.")
            print("TurtleNeck Score = ", tutleneck_score)
            # win10toast 알림 제공
            # toaster.show_toast("TurtleNect WARNING", f"Keep your posture straight.\n\nDegree Of TurtleNeck = {tutleneck_score}")
            # 알림 제공 후 카운트를 다시 0으로 만든다.
            turtle_neck_count = 0

    for face in faces:
        shapes = predictor(gray, face)
        shapes = face_utils.shape_to_np(shapes)

        eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
        eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

        eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
        eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
        eye_img_r = cv2.flip(eye_img_r, flipCode=1)

        # cv2.imshow('l', eye_img_l)
        # cv2.imshow('r', eye_img_r)

        eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
        eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

        pred_l = model.predict(eye_input_l)
        pred_r = model.predict(eye_input_r)

        # visualize
        # state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
        # state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'

        # state_l = state_l % pred_l
        # state_r = state_r % pred_r

        cv2.putText(img, "left"+str(int(np.round(pred_l)[0][0])), (320,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)
        cv2.putText(img, "right"+str(int(np.round(pred_r)[0][0])), (460,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)

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
    