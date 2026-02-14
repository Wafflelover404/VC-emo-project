import cv2

def test_camera():
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press ctrl + c to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera not working :( ")
            break
        cv2.imshow('Camera Test', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()
