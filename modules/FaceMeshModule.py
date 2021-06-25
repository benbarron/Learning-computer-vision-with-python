import cv2
import mediapipe as mp
import time
import numpy


class FaceMesh:
    def __init__(self, maxFaces=1):
        self.faceMesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=maxFaces)
        self.drawSpec = mp.solutions.drawing_utils.DrawingSpec(
            thickness=1, circle_radius=2)
        self.image = None
        self.results = None

    def process(self, image):
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.image)

    def foundFace(self):
        if not self.results:
            return False
        if not self.results.multi_face_landmarks:
            return False
        return True

    def getPoints(self):
        points = []
        if self.results.multi_face_landmarks:
            for lms in self.results.multi_face_landmarks:
                for id, lm in enumerate(lms.landmark):
                    height, width, chan = self.image.shape
                    x, y = int(lm.x * width), int(lm.y * height)
                    points.append((id, x, y))
        return points

    def computeCentroid(self, points=None):
        if points == None:
            points = self.getPoints()
        xmedian = numpy.median([p[1] for p in points])
        ymedian = numpy.median([p[2] for p in points])
        return int(xmedian), int(ymedian)

    def drawFace(self):
        drawSpec = mp.solutions.drawing_utils.DrawingSpec(
            thickness=1, circle_radius=2)
        if self.results.multi_face_landmarks:
            for lms in self.results.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    self.image, lms, mp.solutions.face_mesh.FACE_CONNECTIONS, drawSpec, drawSpec)

    def getImage(self):
        return cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)


def main():
    cap = cv2.VideoCapture(0)
    detector = FaceMesh()

    while cv2.waitKey(1) != ord('q'):
        hasFrame, frame = cap.read()
        detector.process(frame)
        detector.drawFace()
        points = detector.getPoints()
        center = detector.computeCentroid(points)
        frame = detector.getImage()
        cv2.circle(frame, center, 2, (255, 0, 0), 5)
        cv2.imshow('image', frame)

    cap.release()


if __name__ == '__main__':
    main()
