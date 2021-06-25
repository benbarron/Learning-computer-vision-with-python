import cv2
import time
import numpy as np
from modules.HandTrackingModule import HandDetector

detector = HandDetector()
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)
prevTime = 0

_, frame = cap.read()
signature = np.ones(frame.shape, dtype='uint8') * 255

prevPosition = None


while cv2.waitKey(1) != ord('q'):
    hasFrame, frame = cap.read()
    detector.findHands(frame)
    detector.drawHands()
    frame = detector.getImage()
    position = detector.getLandmarkPosition(8)
    cursor = cv2.circle(signature.copy(), position, 5, (255, 0, 0), 3)
    cursor = cv2.flip(cursor, 1)

    if prevPosition != None and position != None:
        position = ((position[0]//10)*10), ((position[1]//10)*10)
        cv2.line(signature, prevPosition, position, (0, 0, 0), 5)

    currentTime = time.time()
    fps = 1 / (currentTime - prevTime)
    prevTime = currentTime
    prevPosition = position
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow('Image', frame)
    cv2.imshow('Signature', cursor)

    if cv2.waitKey(1) == ord('c'):
        signature = np.ones(frame.shape, dtype='uint8') * 255

cap.release()
