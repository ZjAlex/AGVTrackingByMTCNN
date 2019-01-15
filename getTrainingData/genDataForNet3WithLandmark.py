#coding:utf-8
import sys
import numpy as np
import cv2
import os
import argparse
import pickle
from toolsFunc.commomUtils import IoU, convert_to_square
from models.nets import Net_1, Net_2, Net_3
from models.configs import config
from toolsFunc.loaderUtils import TestLoader
from detectors.net1Detector import Net1Detector
from detectors.net2Detector import Net2Detector
from detectors.net3Detector import Net3Detector
from detectors.wholeNetsDetector import WholeDetector

rootPath = "D:/alex/CascadeNetwork3/"


def read_annotation(ImagesPath, annoTxtPath):
    data = dict()
    images = []
    bboxes = []
    labelfile = open(annoTxtPath, 'r')
    while True:
        # image path
        # image path
        line = labelfile.readline().split(' ')
        imagepath = line[0]
        if not imagepath:
            break
        imagepath = os.path.join(ImagesPath, imagepath)
        images.append(imagepath)
        # face numbers
        one_image_bboxes = []
        # only need x, y, w, h
        xmin = float(line[1])
        ymin = float(line[2])
        xmax = xmin + float(line[3])
        ymax = ymin + float(line[4])
        one_image_bboxes.append([xmin, ymin, xmax, ymax])
        bboxes.append(one_image_bboxes)
    data['images'] = images#all image pathes
    data['bboxes'] = bboxes#all image bboxes
    return data


def read_annotation_v2(ImagesPath, annoTxtPath):
    data = dict()
    images = []
    bboxes = []
    landmarks = []
    labelfile = open(annoTxtPath, 'r')
    while True:
        # image path
        # image path
        line = labelfile.readline().split(' ')
        imagepath = line[0]
        if not imagepath:
            break
        imagepath = os.path.join(ImagesPath, imagepath)
        images.append(imagepath)
        # object numbers
        one_image_bboxes = []
        one_image_landmarks = []
        # only need x, y, w, h
        xmin = float(line[1])
        ymin = float(line[2])
        xmax = xmin + float(line[3])
        ymax = ymin + float(line[4])
        one_image_bboxes.append([xmin, ymin, xmax, ymax])

        x_point1 = float(line[5])
        y_point1 = float(line[6])
        x_point2 = float(line[7])
        y_point2 = float(line[8])
        one_image_landmarks.append([x_point1, y_point1, x_point2, y_point2])

        bboxes.append(one_image_bboxes)
        landmarks.append(one_image_landmarks)
    data['images'] = images#all image pathes
    data['bboxes'] = bboxes#all image bboxes
    data['landmarks'] = landmarks#all image landmarks
    return data


def __save_data(stage, data, save_path):
    im_idx_list = data['images']
    gt_boxes_list = data['bboxes']
    gt_lms_list = data['landmarks']
    num_of_images = len(im_idx_list)
    # save files
    saveFolder = os.path.join(rootPath, "tmp/data/%s/"%(stage))
    print(">>>>>> Gen hard samples for %s..."%(stage))
    typeName = ["pos"]
    saveFiles = {}
    for tp in typeName:
        _saveFolder = os.path.join(saveFolder, tp)
        if not os.path.isdir(_saveFolder):
            os.makedirs(_saveFolder)
        saveFiles[tp] = open(os.path.join(saveFolder, "{}.txt".format(tp)), 'w')
    #read detect result
    det_boxes = pickle.load(open(os.path.join(save_path, 'detections.pkl'), 'rb'))
    assert len(det_boxes) == num_of_images, "incorrect detections or ground truths"
    # index of neg, pos and part face, used as their image names
    p_idx = 0
    total_idx = 0
    total_num = 0
    for im_idx, dets, gts, gt_lm in zip(im_idx_list, det_boxes, gt_boxes_list, gt_lms_list):
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
       # print(dets.shape[0])
        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)
        total_idx += 1
        total_num += dets.shape[0]
        #change to square
        #dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        max_iou = 0
        box_id = -1
        for i, box in enumerate(dets):
            iou = IoU(box, gts)
            if iou > max_iou:
                max_iou = iou
                box_id = i

        for i, box in enumerate(dets):
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1
            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue
            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            image_size = 48
            resized_im = cv2.resize(cropped_im, (image_size * 2, image_size),
                                    interpolation=cv2.INTER_LINEAR)


            # find gt_box with the highest iou
            idx = np.argmax(Iou)
            assigned_gt = gts[idx]
            x1, y1, x2, y2 = assigned_gt
            x_point1, y_point1, x_point2, y_point2 = gt_lm[idx]
            # compute bbox reg label
            offset_x1 = (x1 - x_left) / float(width)
            offset_y1 = (y1 - y_top) / float(height)
            offset_x2 = (x2 - x_right) / float(width)
            offset_y2 = (y2 - y_bottom) / float(height)

            offset_x_point1 = (x_point1 - x_left) / float(width)
            offset_x_point2 = (x_point2 - x_left) / float(width)
            offset_y_point1 = (y_point1 - y_top) / float(height)
            offset_y_point2 = (y_point2 - y_top) / float(height)
            # save positive and part-face images and write labels
            if np.max(Iou) >= 0.8 or i == box_id:
                save_file = os.path.join(saveFolder, "pos", "%s.jpg"%p_idx)
                saveFiles['pos'].write(save_file + ' 1 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2, offset_x_point1, offset_y_point1, offset_x_point2, offset_y_point2))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1

        printStr = "\r[{}] pos: {}".format(total_idx, p_idx)
        sys.stdout.write(printStr)
        sys.stdout.flush()
    for f in saveFiles.values():
        f.close()
    print(total_num)
    print('\n')

def test_net(batch_size, stage, thresh, min_face_size, stride):
    print(">>>>>> Detect bbox for %s..."%(stage))
    detectors = [None, None, None]
    if stage in ["net3_v2"]:
        modelPath = os.path.join(rootPath, 'tmp/model/net1/')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('net1-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "net1-%d"%(maxEpoch))
        print("Use Net1 model: %s"%(modelPath))
        Net1 = Net1Detector(Net_1, modelPath)
        detectors[0] = Net1
    if stage in ["net3_v2"]:
        modelPath = os.path.join(rootPath, 'tmp/model/net2/')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('net2-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "net2-%d"%(maxEpoch))
        print("Use Net2 model: %s"%(modelPath))
        Net2 = Net2Detector(Net_2, 24, batch_size, modelPath)
        detectors[1] = Net2
    # read annatation(type:dict)
    ImagesPath = 'D:/alex/CascadeNetwork3'
    annoTxtPath = 'D:/alex/CascadeNetwork3/images1/gt.txt'
    data = read_annotation_v2(ImagesPath, annoTxtPath)
    wholeDetector = WholeDetector(detectors=detectors, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh)
    test_data = TestLoader(data['images'])
    # do detect
    detections, _ = wholeDetector.detect_face(test_data)
    # save detect result
    save_path = os.path.join(rootPath, "tmp/data", stage)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, "detections.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(detections, f, 1)
    print("\nDone! Start to do OHEM...")
    __save_data(stage, data, save_path)


if __name__ == '__main__':
    stage = "net3_v2"

    batchSize = 1
    threshold = [0.4, 0.05]
    minFace = 48
    stride = 2
    test_net(
          batchSize, #test batch_size
          stage, # net3
          threshold, #cls threshold
          minFace, #min_face
          stride)
