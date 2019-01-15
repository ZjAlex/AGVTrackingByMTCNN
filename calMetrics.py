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
import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
            cv2.putText(image,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
            cv2.rectangle(image, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))
        allLandmark = allLandmarks[idx]
        if allLandmark is not None: # pnet and rnet will be ignore landmark
            for landmark in allLandmark:
                for i in range(int(len(landmark)/2)):
                    cv2.circle(image, (int(landmark[2*i]),int(int(landmark[2*i+1]))), 3, (0,0,255))
        savePath = os.path.join(rootPath, 'testing', 'results_%s'%(stage))
        if not os.path.isdir(savePath):
            os.makedirs(savePath)
        cv2.imwrite(os.path.join(savePath, "result_%d.jpg" %(idx)), image)
        print("Save image to %s"%(savePath))


def calc_metrics(stage):
    #print("Start testing in %s" % (testFolder))
    detectors = [None, None, None]
    if stage in ['net1', 'net2', 'net2_v2', 'net3', 'net3_v2']:
        modelPath = os.path.join(rootPath, 'tmp/model/net1/')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('net1-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))  # auto match a max epoch model
        modelPath = os.path.join(modelPath, "net1-%d" % (maxEpoch))
        print("Use Net1 model: %s" % (modelPath))
        detectors[0] = Net1Detector(Net_1, modelPath)
    if stage in ['net2', 'net3', 'net3_v2']:
        modelPath = os.path.join(rootPath, 'tmp/model/net2/')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('net2-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "net2-%d" % (maxEpoch))
        print("Use Net2 model: %s" % (modelPath))
        detectors[1] = Net2Detector(Net_2, 24, 1, modelPath)

    if stage in ['net2_v2']:
        modelPath = os.path.join(rootPath, 'tmp/model/net2_v2/')
        a = [b[8:-6] for b in os.listdir(modelPath) if b.startswith('net2_v2-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "net2_v2-%d" % (maxEpoch))
        print("Use Net2_v2 model: %s" % (modelPath))
        detectors[1] = Net2_V2Detector(Net_2_v2, 24, 1, modelPath)

    if stage in ['net3']:
        modelPath = os.path.join(rootPath, 'tmp/model/net3/')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('net3-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "net3-%d" % (maxEpoch))
        print("Use Net3 model: %s" % (modelPath))
        detectors[2] = Net3Detector(Net_3, 48, 1, modelPath)
    if stage in ['net3_v2']:
        modelPath = os.path.join(rootPath, 'tmp/model/net3_v2/')
        a = [b[8:-6] for b in os.listdir(modelPath) if b.startswith('net3_v2-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "net3_v2-%d" % (maxEpoch))
        print("Use Net3_v2 model: %s" % (modelPath))
        detectors[2] = Net3Detector(Net_3_V2, 48, 1, modelPath)
    wholeDetector = WholeDetector(detectors=detectors, min_face_size=48, threshold=[0.7, 0.8, 0.0])

    with open("D:/alex/CascadeNetwork3/images1/gt.txt") as f:
        gt_lines = f.readlines()

    tp = 0
    fp = 0
    total_true = 0
    iou_sum = 0
    left_x_pixel_offset = 0
    left_y_pixel_offset = 0
    right_x_pixel_offset = 0
    right_y_pixel_offset = 0
    net1_inference_time = 0
    net2_inference_time = 0
    net3_inference_time = 0
    pred_prob = 0

    divide_nums = 0

    save_path = "D:/alex/CascadeNetwork3/images1/res_1.txt"
    res_f = open(save_path, 'w')

    for line in gt_lines:
        line = line.split(' ')
        image_path = line[0]
        if image_path.split('/')[1] != '1':
            continue
        x1 = float(line[1])
        y1 = float(line[2])
        x2 = x1 + float(line[3])
        y2 = y1 + float(line[4])
        lm1_x = float(line[5])
        lm1_y = float(line[6])
        lm2_x = float(line[7])
        lm2_y = float(line[8])
        total_true += 1

        img = cv2.imread(rootPath + image_path)
        all_boxes, landmarks, inf1, inf2, inf3, pred_prob = wholeDetector.detect_face_oneImage(img)
        res_f.write('pred_prob: ' + str(pred_prob) + '\n')
        print(inf1)
        if len(all_boxes) == 0 or len(all_boxes[0]) == 0:
            continue
        tp += 1
        if len(all_boxes) > 1:
            fp += len(all_boxes) - 1
            continue
        divide_nums += 1
        iou = calc_iou([x1, y1, x2, y2], all_boxes[0])
        iou_sum += iou
        net1_inference_time += inf1
        net2_inference_time += inf2
        net3_inference_time += inf3
        left_x_pixel_offset += abs(landmarks[0][0][0] - lm1_x)
        left_y_pixel_offset += abs(landmarks[0][0][1] - lm1_y)
        right_x_pixel_offset += abs(landmarks[0][0][2] - lm2_x)
        right_y_pixel_offset += abs(landmarks[0][0][3] - lm2_y)
        res_f.write('pred_prob: ' + str(pred_prob) + '\n')

    '''res_f.write('tp: ' + str(tp) + '\n')
    res_f.write('fp: ' + str(fp) + '\n')
    res_f.write('total_true: ' + str(total_true) + '\n')
    res_f.write('iou_sum: ' + str(iou_sum) + '\n')
    res_f.write('left_x_pixel_offset: ' + str(left_x_pixel_offset) + '\n')
    res_f.write('left_y_pixel_offset: ' + str(left_y_pixel_offset) + '\n')
    res_f.write('right_x_pixel_offset: ' + str(right_x_pixel_offset) + '\n')
    res_f.write('right_y_pixel_offset: ' + str(right_y_pixel_offset) + '\n')
    res_f.write('net1_inference_time: ' + str(net1_inference_time) + '\n')
    res_f.write('net2_inference_time: ' + str(net2_inference_time) + '\n')
    res_f.write('net3_inference_time: ' + str(net3_inference_time) + '\n')
    res_f.write('divide_nums: ' + str(divide_nums) + '\n')'''

    res_f.close()

def calc_iou(a, b):
    a_x1 = a[0]
    a_y1 = a[1]
    a_x2 = a[2]
    a_y2 = a[3]

    b_x1 = b[0][0]
    b_y1 = b[0][1]
    b_x2 = b[0][2]
    b_y2 = b[0][3]

    area_a = max(0, a_x2 - a_x1) * max(0, a_y2 - a_y1)
    area_b = max(0, b_x2 - b_x1) * max(0, b_y2 - b_y1)
    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    return (inter_area) / (area_a + area_b - inter_area)

def draw_pre_reca():
    save_path = "D:/alex/CascadeNetwork3/images1/res_1.txt"
    with open(save_path, 'r') as f:
        lines = f.readlines()
    probs = []
    for i in range(len(lines)):
        probs.append(float(lines[i][11:]))
    probs = sorted(probs, reverse=True)
    x = []
    y = []
    y_sum = 0
    tp = 0
    fp = 0
    num = len(probs)
    total_true = 0
    for i in range(num):
        if i != 0:
            total_true += 1
    for i in range(num):
        if probs[i] != 0:
            tp += 1
            x.append(tp / (tp + fp))
            y.append(tp / total_true)
        else:
            print(0)
            fp += 1
            x.append(tp / (tp + fp))
            y.append(tp / total_true)

    plt.scatter(y, x)
    plt.show()


if __name__ == "__main__":
    stage = 'net3'
    if stage not in ['net1', 'net2', 'net2_v2', 'net3', 'net3_v2']:
        raise Exception("Please specify stage by --stage=pnet or rnet or onet")
    #calc_metrics(stage)
    draw_pre_reca()


