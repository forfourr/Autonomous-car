import time
import picamera

def main():
    try:
        # 카메라 초기화
        with picamera.PiCamera() as camera:
            # 카메라 해상도 설정 (여기에서는 640x480)
            camera.resolution = (640, 480)
            # 카메라 프리뷰 시작
            camera.start_preview()

            # 카메라 미리보기가 실행되는 동안 10초간 대기
            time.sleep(10)

            # 카메라 미리보기 중지
            camera.stop_preview()

    except KeyboardInterrupt:
        print("카메라 미리보기를 종료합니다.")

if __name__ == "__main__":
    main()
