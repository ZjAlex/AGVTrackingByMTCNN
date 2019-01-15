#coding:utf-8
import tensorflow as tf
import numpy as np
import os
import sys
from models.nets import Net_1, Net_2, Net_3, Net_3_V2, Net_2_v2
from toolsFunc.loaderUtils import TestLoader
from detectors.wholeNetsDetector import WholeDetector
from detectors.net1Detector import Net1Detector
from detectors.net2Detector import Net2Detector
from detectors.net3Detector import Net3Detector
from detectors.net2_v2Detector import Net2_V2Detector
import cv2
import argparse
import time

rootPath = "D:/alex/CascadeNetwork3/"


def test(stage, testFolder):
    print("Start testing in %s"%(testFolder))
    detectors = [None, None, None]
    if stage in ['net1', 'net2', 'net2_v2', 'net3', 'net3_v2']:
        modelPath = os.path.join(rootPath, 'tmp/model/net1/')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('net1-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))# auto match a max epoch model
        modelPath = os.path.join(modelPath, "net1-%d"%(maxEpoch))
        print("Use Net1 model: %s"%(modelPath))
        detectors[0] = Net1Detector(Net_1, modelPath)
    if stage in ['net2', 'net3', 'net3_v2']:
        modelPath = os.path.join(rootPath, 'tmp/model/net2/')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('net2-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "net2-%d"%(maxEpoch))
        print("Use Net2 model: %s"%(modelPath))
        detectors[1] = Net2Detector(Net_2, 24, 1, modelPath)

    if stage in ['net2_v2']:
        modelPath = os.path.join(rootPath, 'tmp/model/net2_v2/')
        a = [b[8:-6] for b in os.listdir(modelPath) if b.startswith('net2_v2-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "net2_v2-%d"%(maxEpoch))
        print("Use Net2_v2 model: %s"%(modelPath))
        detectors[1] = Net2_V2Detector(Net_2_v2, 24, 1, modelPath)

    if stage in ['net3']:
        modelPath = os.path.join(rootPath, 'tmp/model/net3/')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('net3-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "net3-%d"%(maxEpoch))
        print("Use Net3 model: %s"%(modelPath))
        detectors[2] = Net3Detector(Net_3, 48, 1, modelPath)
    if stage in ['net3_v2']:
        modelPath = os.path.join(rootPath, 'tmp/model/net3_v2/')
        a = [b[8:-6] for b in os.listdir(modelPath) if b.startswith('net3_v2-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "net3_v2-%d" % (maxEpoch))
        print("Use Net3_v2 model: %s" % (modelPath))
        detectors[2] = Net3Detector(Net_3_V2, 48, 1, modelPath)
    wholeDetector = WholeDetector(detectors=detectors, min_face_size=48, threshold=[0.7, 0.8, 0.0])

    testImages = []
    for name in os.listdir(testFolder):
        testImages.append(os.path.join(testFolder, name))
    testDatas = TestLoader(testImages)
    # Now to detect
    allBoxes, allLandmarks = wholeDetector.detect_face(testDatas)

    print("\n")
    # Save it
    for idx, imagePath in enumerate(testImages):
        image = cv2.imread(imagePath)
        for bbox in allBoxes[idx]:
            cv2.putText(image,str(0.98),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,4,color=(255,0,255),thickness=4)
            cv2.rectangle(image, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255), 5)
        allLandmark = allLandmarks[idx]
        if allLandmark is not None: # pnet and rnet will be ignore landmark
            for landmark in allLandmark:
                for i in range(int(len(landmark)/2)):
                    cv2.circle(image, (int(landmark[2*i]),int(int(landmark[2*i+1]))), 3, (0,0,255), thickness=3)
        savePath = os.path.join(rootPath, 'testing', 'results_%s'%(stage))
        if not os.path.isdir(savePath):
            os.makedirs(savePath)
        cv2.imwrite(os.path.join(savePath, "result_%d.jpg" %(idx)), image)
        print("Save image to %s"%(savePath))


if __name__ == "__main__":
    stage = 'net3'
    if stage not in ['net1', 'net2', 'net2_v2', 'net3', 'net3_v2']:
        raise Exception("Please specify stage by --stage=pnet or rnet or onet")
    test(stage, os.path.join(rootPath, "testing", "images"))

