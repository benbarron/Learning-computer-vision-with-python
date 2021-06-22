import cv2
import time
from modules.HandTrackingModule import HandDetector

detector = HandDetector()
cap = cv2.VideoCapture(1)
prevTime = 0

comparisons = [
    (4, 5),
    (8, 6),
    (12, 10),
    (16, 14),
    (20, 18)
]

while cv2.waitKey(1) != ord('q'):
    hasFrame, frame = cap.read()

    detector.findHands(frame)
    detector.drawHands()
    frame = detector.getImage()

    counter = 0
    for comp in comparisons:
        pos1 = detector.getLandmarkPosition(comp[0])
        pos2 = detector.getLandmarkPosition(comp[1])
        if pos1 != None and pos2 != None:
            if pos1[1] < pos2[1]:
                counter += 1

    print(counter)

    currentTime = time.time()
    fps = 1 / (currentTime - prevTime)
    prevTime = currentTime

    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", frame)

cap.release()
