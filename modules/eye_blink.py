from win10toast import ToastNotifier
import math


eye_blink_count = 0

eye_blinker = 300

def eyeblink_detection(detector, img, sensitivity, log=False, notification=True):

    # toast 알림을 주는 객체 생성
    eyeblink_toaster = ToastNotifier()

    global eye_blink_count
    global eye_blinker

    right_eye_length, img = detector.findEyeBlink(159, 145, img, draw=True, r=5, t=2)
    left_eye_length, img = detector.findEyeBlink(386, 374, img, draw=True, r=5, t=2)
    
    eye_depth = detector.findEyeDepth(8, 9)
    
    right_eye_length = right_eye_length*4
    left_eye_length = left_eye_length*4

    if eye_depth > 6:
        eye_blink_threshold = round(abs(math.log2(eye_depth)) * sensitivity)
    else:
        eye_blink_threshold = 15

    # eye_blink_threshold = eye_depth
    # eye_blink_threshold = np.interp(eye_blink_threshold, [0, 15], [0, 80])

    if right_eye_length < eye_blink_threshold and left_eye_length < eye_blink_threshold:
        eye_blink_count += 1
        eye_blinker = 300

    eye_blinker -= 1
    if eye_blinker < 0:
        if notification:
            eyeblink_toaster.show_toast("Dry eyes WARNING", f" \nPlease blink your eyes.\n")
        eye_blinker = 300

    if log:
        print("length : ", round(right_eye_length), "  " ,round(left_eye_length), "eye_blink_threshold :", round(eye_blink_threshold, 2), "eye_blink_count :", eye_blink_count)
        # print("right_eye_length : ", round(right_eye_length**2), "left_eye_length : ", round(left_eye_length**2), "eye_blink_threshold :", round(eye_blink_threshold, 2), "eye_blink_count :", eye_blink_count)