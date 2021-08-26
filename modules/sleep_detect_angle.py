from win10toast import ToastNotifier
import math

sleep_count = 0

def sleepiness_detection(detector, img, log=False, notification=True):

    global sleep_count

    # toast 알림을 주는 객체 생성
    sleep_detection_toaster = ToastNotifier()
    
    face_lmList = detector.findFaceLandmark(img, draw=False)

    if len(face_lmList) != 0:
        x1, y1 = face_lmList[33][1:3]
        x2, y2 = face_lmList[159][1:3]
        x3, y3 = face_lmList[145][1:3] 
        
        angle_1 = math.atan(abs(y2-y1)/abs(x2-x1+0.01)) 
        angle_2 = math.atan(abs(y3-y1)/abs(x3-x1+0.01)) 

        x4, y4 = face_lmList[263][1:3]
        x5, y5 = face_lmList[386][1:3]
        x6, y6 = face_lmList[374][1:3] 
        
        angle_3 = math.atan(abs(y5-y4)/abs(x5-x4+0.01)) 
        angle_4 = math.atan(abs(y6-y4)/abs(x6-x4+0.01)) 

        total_angle = math.degrees(angle_1 + angle_2 + angle_3 + angle_4) ** 2
        
        
    if total_angle < 5000:
        sleep_count += 1
    else:
        sleep_count = 0

    
    if sleep_count > 200:
        if notification:
            sleep_detection_toaster.show_toast("Sleepiness WARNING", f" \nPlease Stretch your body.\n")
        print("[ Sleepiness WARNING ] - Please Stretch your body")
        sleep_count = 0

    if log:
        print("angle :", round(total_angle), "  sleep_count :", sleep_count)#, "  close_eye_threshold :", close_eye_threshold)
