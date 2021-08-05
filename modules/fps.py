import time
import cv2

# privious time for fps
pTime = 0
# cerrent time for fps
cTime = 0

def fps_present(img, draw=True):
    global pTime
    # fps 계산 로직
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    if draw:
        # fps를 이미지 상단에 입력하는 로직
        cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)