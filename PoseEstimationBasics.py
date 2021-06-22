import cv2
import mediapipe as mp
import time

pose = mp.solutions.pose.Pose()
cap = cv2.VideoCapture(1)
prevTime = 0

while cv2.waitKey(1) != ord('q'):
    hasImage, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)

    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

    currentTime = time.time()
    fps = 1 / (currentTime - prevTime)
    prevTime = currentTime

    cv2.putText(image, str(int(fps)), (70, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow('image', image)
    cv2.waitKey(1)

cap.release()
