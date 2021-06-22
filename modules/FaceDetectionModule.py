import cv2
import mediapipe as mp
import time


class FaceDetector:

    def __init__(self):
        self.detector = mp.solutions.face_detection.FaceDetection()
        self.results = None
        self.image = None
        self.faces = None

    def findFaces(self, image):
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.detector.process(self.image)
        faces = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                faceBox = detection.location_data.relative_bounding_box
                height, width, chan = self.image.shape
                p1 = int(faceBox.xmin * width)
                p2 = int(faceBox.ymin * height)
                p3 = int(faceBox.width * width)
                p4 = int(faceBox.height * height)
                faces.append((id, detection.score[0], p1, p2, p3, p4))
        self.faces = faces

    def drawFaces(self):
        if self.faces:
            for face in self.faces:
                cv2.rectangle(self.image, face[2:], (255, 0, 255), 2)
                cv2.putText(self.image, f'{int(face[1]*100)}%',
                            (face[2], face[3] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    def getImage(self):
        return cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)


def main():
    cap = cv2.VideoCapture(1)
    prevTime = 0

    detector = FaceDetector()

    while cv2.waitKey(1) != ord('q'):
        hasFrame, frame = cap.read()
        detector.findFaces(frame)
        detector.drawFaces()
        frame = detector.getImage()

        currentTime = time.time()
        fps = 1 / (currentTime - prevTime)
        prevTime = currentTime

        cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 2)

        cv2.imshow('image', frame)
    cap.release()


if __name__ == '__main__':
    main()
