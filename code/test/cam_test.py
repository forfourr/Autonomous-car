import cv2
import time
import sys
from Servo import Servo
from front_wheel_curve_test import Front_wheels
# 카메라 열기
cap = cv2.VideoCapture(-1)

# 비디오 코덱 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/pi/AI-self-driving-RC-car/code/test/data/tmp/test_record3.avi', fourcc, 20.0, (640, 480))
# stereo 설정
a = Servo(1)
a.setup()
a.write(90)

# 주행

front_wheels = Front_wheels(channel=0)

front_wheels.to90()



while cap.isOpened():
    start = time.time()
    try:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)

        elapse = time.time() - start
        fps = 1/elapse
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except:
        break


    
cap.release()
out.release()
cv2.destroyAllWindows()
