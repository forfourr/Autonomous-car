import cv2

def main():
    # 웹캠 연결
    cap = cv2.VideoCapture(0)

    # 웹캠이 정상적으로 열렸는지 확인
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    while True:
        # 프레임 읽기
        ret, frame = cap.read()

        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # 화면에 프레임 출력
        cv2.imshow('Webcam', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
