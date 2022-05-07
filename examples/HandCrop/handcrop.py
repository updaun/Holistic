import cv2
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import modules.HolisticModule as hm

# video input 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Holistic 객체 생성
detector = hm.HolisticDetector()


while True:

    success, img = cap.read()

    # mediapipe를 거친 이미지 생성 -> img
    img = detector.findHolistic(img, draw=False)
    lmList = detector.findPoseLandmark(img, draw=False)
    h, w, c = img.shape

    if len(lmList) != 0:
        right_index = lmList[20][1:]
        right_pinky = lmList[18][1:]
        cx, cy = (right_index[0] + right_pinky[0])//2, (right_index[1] + right_pinky[1])//2
        padding = 64
        color = (255, 0, 0)

        if padding < cx < w-padding and padding < cy < h-padding:

            img = cv2.rectangle(img, (cx-padding, cy-padding), (cx+padding, cy+padding), color, 2)
            hand_crop = img[cy-padding:cy+padding, cx-padding:cx+padding]
            cv2.imshow("handcrop", hand_crop)

   
    # img를 우리에게 보여주는 부분
    cv2.imshow("Image", img)
    

    # ESC 키를 눌렀을 때 창을 모두 종료하는 부분
    if cv2.waitKey(1) & 0xFF == 27:
        break 

cap.release()
cv2.destroyAllWindows()
    