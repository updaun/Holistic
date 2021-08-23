import cv2
import time
import modules.HolisticModule as hm
from win10toast import ToastNotifier
import math

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


while True:
    # defalut BGR img
    success, img = cap.read()

    # mediapipe를 거친 이미지 생성 -> img
    img = detector.findHolistic(img, draw=True)

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
            toaster.show_toast("TurtleNect WARNING", f"Keep your posture straight.\n\nDegree Of TurtleNeck = {tutleneck_score}")
            # 알림 제공 후 카운트를 다시 0으로 만든다.
            turtle_neck_count = 0

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
    