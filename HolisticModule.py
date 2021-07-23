import cv2
import mediapipe as mp
import time
import math

class HolisticDetector():
    def __init__(self,
               static_image_mode=False,
               model_complexity=1,
               smooth_landmarks=True,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHolistic = mp.solutions.holistic
        self.holistics = self.mpHolistic.Holistic(self.static_image_mode, self.model_complexity, self.smooth_landmarks, self.min_detection_confidence, self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

        # self.tipIds = [4, 8, 12, 16, 20]

    def findHolistic(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
        self.results = self.holistics.process(imgRGB)

        if self.results.pose_landmarks.landmark:
            for holisticLms in self.results.pose_landmarks.landmark:
                if draw :
                    self.mpDraw.draw_landmarks(img, holisticLms, self.mpHolistic.POSE_CONNECTIONS)
        return img

#     def findPosition(self, img, handNo=0, draw=True):
#         xList = []
#         yList = []
#         bbox = []
#         self.lmList = []
#         if self.results.multi_hand_landmarks:
#             myHand = self.results.multi_hand_landmarks[handNo]

#             for id, lm in enumerate(myHand.landmark):
#                 #print(id,lm)
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x*w), int(lm.y*h)
#                 #print(id, cx, cy)
#                 xList.append(cx)
#                 yList.append(cy)
#                 self.lmList.append([id, cx, cy])
#                 #if id == 4:
#                 if draw:
#                     cv2.circle(img, (cx,cy), 7, (255,0,0),cv2.FILLED) 

#             xmin, xmax = min(xList), max(xList)
#             ymin, ymax = min(yList), max(yList)
#             bbox = xmin, ymin, xmax, ymax

#             if draw:
#                 cv2.rectangle(img, (xmin-20, ymin-20), (xmax + 20, ymax + 20),
#                 (0,255,0), 2)

#         return self.lmList, bbox

#     def fingersUp(self):
#         fingers = []
#         # Thumb(Doesn't filp)
#         # if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
#         # Thumb(Filp)
#         if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
#             fingers.append(1)
#         else:
#             fingers.append(0)

#         # Fingers except Thumb
#         for id in range(1, 5):
#             if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
#                 fingers.append(1)
#             else:
#                 fingers.append(0)

#         return fingers

#     def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
#         x1, y1 = self.lmList[p1][1:]
#         x2, y2 = self.lmList[p2][1:]
#         cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

#         if draw:
#             cv2.line(img, (x1, y1), (x2, y2), (255,0,255), t)
#             cv2.circle(img, (x1, y1), r, (255,0,255), cv2.FILLED)
#             cv2.circle(img, (x2, y2), r, (255,0,255), cv2.FILLED)
#             cv2.circle(img, (cx, cy), r, (0,0,255), cv2.FILLED)
#         length = math.hypot(x2-x1, y2-y1)

#         return length, img, [x1, y1, x2, y2, cx, cy]

# def main():
#     pTime = 0
#     cTime = 0
#     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#     detector = handDetector()
#     while True:
#         success, img = cap.read()
#         img = detector.findHands(img)
#         lmList = detector.findPosition(img)
#         if len(lmList) != 0:
#             print(lmList[4])


#         cTime = time.time()
#         fps = 1/(cTime-pTime)
#         pTime = cTime

#         cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

#         cv2.imshow("Image", img)
#         if cv2.waitKey(1) & 0xFF == 27:
#             break 

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()