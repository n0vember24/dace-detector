import cv2
from mtcnn import MTCNN


def detect_faces(frame, detector: MTCNN):
    result = detector.detect_faces(frame)
    for face in result:
        x, y, w, h = face["box"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame


def main():
    detector = MTCNN()
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        print(frame)
        frame = detect_faces(frame, detector)
        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
