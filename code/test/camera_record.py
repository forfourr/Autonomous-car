import cv2
import time
import sys
from Servo import Servo
from back_wheels import Back_Wheels
from front_wheel_curve_test import Front_wheels


# 카메라 열기
cap = cv2.VideoCapture(0)

# 비디오 코덱 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('captured_video1.avi', fourcc, 20.0, (320, 240))

# stereo 설정
a = Servo(1)
a.setup()
a.write(90)

# 주행
back_wheels = Back_Wheels()
front_wheels = Front_wheels(channel=0)

front_wheels.turning_offset = 90

while cap.isOpened():
    start_time = time.time()

    try:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        
        _time = time.time() - start_time
        fps = 1/_time
        cv2.putText(frame, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        print('fail') 
        break

    

back_wheels.speed =0    
cap.release()
out.release()
cv2.destroyAllWindows()
