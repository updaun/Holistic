from win10toast import ToastNotifier
import math


sleep_count = 0

def sleepiness_detection(detector, img, sensitivity, log=False, notification=True):

    # toast 알림을 주는 객체 생성
    sleep_detection_toaster = ToastNotifier()

    global sleep_count
    # global sleep_detector

    right_eye_length, img = detector.findEyeBlink(159, 145, img, draw=True, r=5, t=2)
    left_eye_length, img = detector.findEyeBlink(386, 374, img, draw=True, r=5, t=2)
    
    eye_depth = detector.findEyeDepth(8, 9)
    
    right_eye_length = right_eye_length*4
    left_eye_length = left_eye_length*4

    if eye_depth > 6:
        eye_blink_threshold = round(abs(math.log2(eye_depth)) * sensitivity)
    else:
        eye_blink_threshold = 15


    if right_eye_length < eye_blink_threshold and left_eye_length < eye_blink_threshold:
        sleep_count += 1
    else:
        sleep_count = 0

    
    if sleep_count > 120:
        if notification:
            sleep_detection_toaster.show_toast("Sleepiness WARNING", f" \nPlease Stretch your body.\n")
        sleep_count = 0

    if log:
        print("length : ", round(right_eye_length), "  " ,round(left_eye_length), "eye_blink_threshold :", round(eye_blink_threshold, 2), "sleep_count :", sleep_count)
        # print("right_eye_length : ", round(right_eye_length**2), "left_eye_length : ", round(left_eye_length**2), "eye_blink_threshold :", round(eye_blink_threshold, 2), "eye_blink_count :", eye_blink_count)