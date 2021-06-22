import mediapipe as mp
import cv2
import time
import math


class HandDetector:

    def __init__(self, mode=False, maxHands=2, detectionConf=0.5, trackingConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackingConf = trackingConf
        self.hands = mp.solutions.hands.Hands(
            mode, maxHands, detectionConf, trackingConf)

    def findHands(self, image, draw=False):
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(self.image)
        if draw:
            self.drawHands()

    def drawHands(self):
        if self.results.multi_hand_landmarks:
            for handLandmark in self.results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    self.image, handLandmark, mp.solutions.hands.HAND_CONNECTIONS)

    def getHandPosition(self, handNumber=0):
        landMarkList = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNumber]
            for id, lm in enumerate(hand.landmark):
                h, w, c = self.image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landMarkList.append((id, cx, cy))
        return landMarkList

    def getLandmarkPosition(self, landMarkNumber, handNumber=0):
        for landmark in self.getHandPosition(handNumber):
            if landmark[0] == landMarkNumber:
                return (landmark[1], landmark[2])
        return None

    def getDistance(self, landMark1, landMark2, handNumber=0):
        lm1, lm2 = None, None
        for landmark in self.getHandPosition(handNumber):
            if landmark[0] == landMark1:
                lm1 = landmark
            if landmark[0] == landMark2:
                lm2 = landmark
        if lm1 == None or lm2 == None:
            raise Exception("LM not found")
        return math.sqrt((lm1[1] - lm2[1])**2 + (lm1[2] - lm2[2])**2)

    def foundHand(self):
        return self.results.multi_hand_landmarks

    def getRGB(self):
        return cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)


def main():
    cap = cv2.VideoCapture(1)
    detector = HandDetector()
    prevTime = 0
    currentTime = 0

    while cv2.waitKey(1) != ord('q'):
        hasFrame, frame = cap.read()
        detector.findHands(frame, draw=True)

        if detector.foundHand():
            print(detector.getDistance(4, 8))

        image = detector.getRGB()

        currentTime = time.time()
        fps = 1 / (currentTime - prevTime)
        prevTime = currentTime

        cv2.putText(image, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)

        cv2.imshow('Image', image)
        cv2.waitKey(1)

    cap.release()


if __name__ == '__main__':
    main()
