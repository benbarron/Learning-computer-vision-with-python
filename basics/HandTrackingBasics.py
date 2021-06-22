import mediapipe as mp
import cv2
import time

cap = cv2.VideoCapture(1)
hands = mp.solutions.hands.Hands()
prevTime = 0
currentTime = 0

while cv2.waitKey(1) != ord('q'):
    hasFrame, frame = cap.read()

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handLandmark.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)

            mp.solutions.drawing_utils.draw_landmarks(
                frame, handLandmark, mp.solutions.hands.HAND_CONNECTIONS)

    currentTime = time.time()
    fps = 1 / (currentTime - prevTime)
    prevTime = currentTime

    cv2.putText(frame, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)

    cv2.imshow('Image', frame)
    cv2.waitKey(1)

cap.release()
