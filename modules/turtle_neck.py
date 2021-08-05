from win10toast import ToastNotifier
import math

turtle_neck_count = 0

def turtlenect_detection(detector, img, sensitivity, log=False, notification=True):

    global turtle_neck_count
    global pose_lmList
    global face_lmList

    # toast 알림을 주는 객체 생성
    turtleneck_toaster = ToastNotifier()

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

    # 핵심 로직 목 길이가 임계치보다 작을 때, 거북목으로 생각한다.
    if length < turtleneck_detect_threshold:
        turtle_neck_count += 1

    # 100번 거북목으로 인식되면 알림을 제공한다. 
    if length < turtleneck_detect_threshold and turtle_neck_count > 100:
        # 얼마나 거북목인지 계산해주는 부분 (0~ 100 점) 
        tutleneck_score = int((turtleneck_detect_threshold - int(length))/turtleneck_detect_threshold*100)
        print("WARNING - Keep your posture straight.")
        print("TurtleNeck Score = ", tutleneck_score)

        if notification:
            # win10toast 알림 제공
            turtleneck_toaster.show_toast("TurtleNect WARNING", f"Keep your posture straight.\n\nDegree Of TurtleNeck = {tutleneck_score}")
        # 알림 제공 후 카운트를 다시 0으로 만든다.
        turtle_neck_count = 0

    if log:
        print("Length : {:.3f},   Threshold : {:.3f},   Pose_depth : {}".format(length, turtleneck_detect_threshold, pose_depth))
