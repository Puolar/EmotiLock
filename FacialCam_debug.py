

import cv2
import time
from deepface import DeepFace


def capture_video(sec=.01):
    #face.env.TF_ENABLE_ONEDNN_OPTS = 0
    cascade_path = cv2.data.haarcascades +  'haarcascade_frontalface_default.xml'

    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise IOError(f"Cannot load face cascade: {cascade_path}")


    # faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)


    if not video.isOpened():
        raise IOError("Cannot open webcam")

    try:
        while video.isOpened():
            time.sleep(sec) 
            success, frame = video.read()
            if not success:
                print("Could not read a frame from the webcam")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, minNeighbors=5)
            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (89, 2, 236), 1)
                face_roi = frame[y:y + h, x:x + w]
                try:
                    analysis = DeepFace.analyze(
                        face_roi,
                        actions=["emotion"],
                        detector_backend="skip",
                        enforce_detection=False,
                        silent=True,
                    )
                    emotion = analysis[0]["dominant_emotion"]
                    cv2.putText(
                        frame,
                        emotion,
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (224, 77, 176),
                        2,
                    )
                    print(emotion)
                except Exception as exc:
                    print(f"Emotion analysis failed: {exc}")

            cv2.imshow("video", frame)
            if cv2.waitKey(1) == ord("q"):
                break
    finally:
        video.release()
        cv2.destroyAllWindows()

capture_video()