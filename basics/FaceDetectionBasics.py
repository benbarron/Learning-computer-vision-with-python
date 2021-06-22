import cv2
import mediapipe as mp
import time

faceDetection = mp.solutions.face_detection.FaceDetection()
cap = cv2.VideoCapture(1)
prevTime = 0

while cv2.waitKey(1) != ord('q'):
    hasFrame, frame = cap.read()
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            faceBox = detection.location_data.relative_bounding_box
            height, width, chan = frame.shape
            p1 = int(faceBox.xmin * width)
            p2 = int(faceBox.ymin * height)
            p3 = int(faceBox.width * width)
            p4 = int(faceBox.height * height)
            cv2.rectangle(frame, (p1, p2, p3, p4), (255, 0, 255), 2)
            cv2.putText(frame, f'{int(detection.score[0]*100)}%',
                        (p1, p2 - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    currentTime = time.time()
    fps = 1 / (currentTime - prevTime)
    prevTime = currentTime

    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 2)

    cv2.imshow('image', frame)


cap.release()
