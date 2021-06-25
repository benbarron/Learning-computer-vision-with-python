from modules.FaceMeshModule import FaceMesh
import matplotlib.pyplot as plt
import face_recognition as fr
import pandas as pd
import numpy as np
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, help='save|compare|show')
parser.add_argument('--name', type=str, help='Name to save/compare face')
parser.add_argument('--source', type=int, help='Camera Source')
args = parser.parse_args()


def getFaceMesh():
    cap = cv2.VideoCapture(0)
    mesh = FaceMesh()
    while not mesh.foundFace():
        hasFrame, frame = cap.read()
        mesh.process(frame)
    cap.release()
    return mesh


def saveFace(x, y, name):
    data = pd.DataFrame(data={'x': x, 'y': y}, columns=['x', 'y'])
    data.to_csv(f'./faces/{name}.csv', header=False)


def readFace(name: str):
    data = pd.read_csv(f'./faces/{name}.csv', header=None)
    return data[1].values, data[2].values


def compareFaces(f1, f2):
    x1, y1 = f1
    x2, y2 = f2
    count = 0
    for i in range(x1.shape[0]-1):
        count += abs(x1[i] - x2[i])
    for i in range(y1.shape[0]-1):
        count += abs(y1[i] - y2[i])
    return count / (x1.shape[0] + y1.shape[0])


def showFace(xOut, yOut):
    fig, ax = plt.subplots(1)
    ax.scatter(xOut, yOut, s=100, marker='.', color='b', linewidths=0.01)
    plt.show()


def getFaceFromCamera():
    mesh = getFaceMesh()
    points = mesh.getPoints()
    center = mesh.computeCentroid(points)
    xOut = np.arange(len(points))
    yOut = np.arange(len(points))
    for i in range(len(points)):
        xOut[i] = center[0] - points[i][1]
        yOut[i] = center[1] - points[i][2]
    return xOut, yOut


if args.mode == 'save':
    face = getFaceFromCamera()
    saveFace(face[0], face[1], args.name)
elif args.mode == 'compare':
    face1 = readFace(args.name)
    face2 = getFaceFromCamera()
    res = compareFaces(face1, face2)
    print(res)
elif args.mode == 'show':
    if args.name:
        face = readFace(args.name)
        showFace(face[0], face[1])
    else:
        face = getFaceFromCamera()
        showFace(face[0], face[1])
