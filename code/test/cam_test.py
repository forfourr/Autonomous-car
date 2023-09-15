import cv2
import time
import sys
from Servo import Servo
from front_wheels import Front_Wheels
# 카메라 열기
cap = cv2.VideoCapture(-1)
time_now= time.time()
# 비디오 코덱 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f'/home/pi/AI-self-driving-RC-car/code/test/data/tmp/obj_source_{time_now}.avi', fourcc, 20.0, (640, 480))

# 카메라 높이
a = Servo(2)
a.setup()
a.write(100)

# 카메라 좌우
b = Servo(1)
b.setup()
b.write(95)

# 주행

front_wheels = Front_Wheels(channel=0)

front_wheels.turn_straight



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
