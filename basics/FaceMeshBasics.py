import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)
prevTime = 0

faceMesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=2)
drawSpec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=2)

while cv2.waitKey(1) != ord('q'):
    hasFrame, frame = cap.read()
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for lms in results.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, lms, mp.solutions.face_mesh.FACE_CONNECTIONS, drawSpec, drawSpec)

            for id, lm in enumerate(lms.landmark):
                height, width, chan = frame.shape
                x, y = int(lm.x * width), int(lm.y * height)
                print(id, x, y)

    currentTime = time.time()
    fps = 1 / (currentTime - prevTime)
    prevTime = currentTime

    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow('image', frame)
    cv2.waitKey(1)

cap.release()
