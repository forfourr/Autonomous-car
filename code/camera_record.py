import cv2
import time
import sys
from Servo import Servo
from back_wheels import Back_Wheels
from front_wheel_curve_test import Front_wheels
import threading


# 카메라 열기
cap = cv2.VideoCapture(0)

# 비디오 코덱 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('captured_video1.avi', fourcc, 20.0, (640, 480))

# stereo 설정
a = Servo(1)
a.setup()
a.write(90)

# 주행
back_wheels = Back_Wheels()
front_wheels = Front_wheels(channel=0)

# back_wheels.speed =0
# exit()
'''def drive():
    for _ in range(2):
        front_wheels.turn_straight()
        back_wheels.backward()
        back_wheels.speed =70
        time.sleep(1.5)

        front_wheels.turn_left()
        back_wheels.backward()
        back_wheels.speed =50
        time.sleep(1.5)


        front_wheels.little_right()
        back_wheels.backward()
        back_wheels.speed =50
        time.sleep(0.5)

        front_wheels.turn_straight()
        back_wheels.backward()
        back_wheels.speed =70
        time.sleep(2.5)'''



while cap.isOpened():
    front_wheels.turn_straight()
    back_wheels.speed = 50

    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

back_wheels.speed =0    
cap.release()
out.release()
cv2.destroyAllWindows()
