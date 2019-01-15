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

def __save_data(stage, data, save_path):
    im_idx_list = data['images']
    gt_boxes_list = data['bboxes']
    num_of_images = len(im_idx_list)
    # save files
    saveFolder = os.path.join(rootPath, "tmp/data/%s/"%(stage))
    print(">>>>>> Gen hard samples for %s..."%(stage))
    typeName = ["pos", "neg", "part"]
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
    n_idx, p_idx, d_idx = 0, 0, 0
    total_idx = 0
    total_num = 0
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
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
        neg_num = 0
        max_iou = 0
        box_id = -1
        # ensure every image has at least one pos
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
            image_size = 24
            resized_im = cv2.resize(cropped_im, (image_size * 2, image_size),
                                    interpolation=cv2.INTER_LINEAR)
            # save negative images and write label
            if np.max(Iou) < 0.60 and neg_num < 60 and i != box_id:
                # now to save it
                save_file = os.path.join(saveFolder, "neg", "%s.jpg"%n_idx)
                saveFiles['neg'].write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            else:
                # find gt_box with the highest iou
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt
                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)
                # save positive and part-face images and write labels
                if np.max(Iou) >= 0.75 or i == box_id:
                    save_file = os.path.join(saveFolder, "pos", "%s.jpg"%p_idx)
                    saveFiles['pos'].write(save_file + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                elif np.max(Iou) >= 0.6:
                    save_file = os.path.join(saveFolder, "part", "%s.jpg"%d_idx)
                    saveFiles['part'].write(save_file + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
        printStr = "\r[{}] pos: {}  neg: {}  part:{}".format(total_idx, p_idx, n_idx, d_idx)
        sys.stdout.write(printStr)
        sys.stdout.flush()
    for f in saveFiles.values():
        f.close()
    print('\n')
    print(total_num)
    print('\n')

def test_net(batch_size, stage, thresh, min_face_size, stride):
    print(">>>>>> Detect bbox for net2")
    detectors = [None, None, None]
    modelPath = os.path.join(rootPath, 'tmp/model/net1/')
    a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('net1-') and b.endswith('.index')]
    maxEpoch = max(map(int, a))
    modelPath = os.path.join(modelPath, "net1-%d"%(maxEpoch))
    print("Use Net1 model: %s"%(modelPath))
    Net1 = Net1Detector(Net_1, modelPath)
    detectors[0] = Net1
    # read annatation(type:dict)
    ImagesPath = 'D:/alex/CascadeNetwork3'
    annoTxtPath = 'D:/alex/CascadeNetwork3/images1/gt.txt'
    data = read_annotation(ImagesPath, annoTxtPath)
    mtcnn_detector = WholeDetector(detectors=detectors, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh)
    test_data = TestLoader(data['images'])
    # do detect
    detections, _ = mtcnn_detector.detect_face(test_data)
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
    stage = 'net2'
    batchSize = 1
    threshold = [0.4, 0.05]
    minFace = 48
    stride = 2
    test_net(
          batchSize, #test batch_size
          stage, # "net2
          threshold, #cls threshold
          minFace, #object min_size
          stride)
