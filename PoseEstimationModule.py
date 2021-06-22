import time
import cv2
import math
import mediapipe as mp


class PoseEstimator:

    def __init__(self):
        self.results = None
        self.image = None
        self.pose = mp.solutions.pose.Pose()

    def findLandmarks(self, image):
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(self.image)

    def drawLandmarks(self):
        if self.results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                self.image, self.results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

    def getAllPositions(self):
        landMarkList = []
        for id, lm in enumerate(self.results.pose_landmarks.landmark):
            h, w, c = self.image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            landMarkList.append((id, cx, cy))
        return landMarkList

    def getLandmarkPosition(self, landmark):
        for lm in self.getAllPositions():
            if lm[0] == landmark:
                return (lm[1], lm[2])
        return None

    def getDistance(self, landMark1, landMark2):
        lm1, lm2 = None, None
        for landmark in self.getAllPositions():
            if landmark[0] == landMark1:
                lm1 = landmark
            if landmark[0] == landMark2:
                lm2 = landmark
        if lm1 == None or lm2 == None:
            raise Exception("LM not found")
        return math.sqrt((lm1[1] - lm2[1])**2 + (lm1[2] - lm2[2])**2)

    def getImage(self):
        return cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)


def main():
    estimator = PoseEstimator()
    cap = cv2.VideoCapture(1)
    prevTime = 0

    while cv2.waitKey(1) != ord('q'):
        hasFrame, frame = cap.read()
        estimator.findLandmarks(frame)
        estimator.drawLandmarks()
        image = estimator.getImage()

        # print(estimator.getLandmarkPosition(1))
        print(estimator.getDistance(20, 19))

        currentTime = time.time()
        fps = 1 / (currentTime - prevTime)
        prevTime = currentTime

        cv2.putText(image, str(int(fps)), (70, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow('image', image)
        cv2.waitKey(1)

    cap.release()


if __name__ == "__main__":
    main()
