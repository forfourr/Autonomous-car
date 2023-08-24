import cv2
import time

# 카메라 열기
cap = cv2.VideoCapture(0)

# 비디오 코덱 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('test_curve.avi', fourcc, 20.0, (320, 240))



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
