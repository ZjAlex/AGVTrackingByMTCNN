# coding: utf-8
import os
import math
from os.path import join, exists
import cv2
import numpy as np
import random
import sys
import argparse
from toolsFunc.commomUtils import getBboxLandmarkFromTxt, IoU, BBox
from toolsFunc.landmarkUtils import rotate, flip

rootPath = "D:/alex/CascadeNetwork3/"


def gen_landmark_data(srcTxt, net, augment=False):
    '''
    srcTxt: each line is: 0=path, 1-4=bbox, 5-14=landmark 5points
    net: PNet or RNet or ONet
    augment: if enable data augmentation
    '''
    print(">>>>>> Start landmark data create...Stage: %s" % (net))
    srcTxt = os.path.join(rootPath, srcTxt)
    saveFolder = os.path.join(rootPath, "tmp/data/%s/" % (net))
    saveImagesFolder = os.path.join(saveFolder, "landmark")
    sizeOfNet = {"net1": 12, "net2_v2": 24, "net3": 48}
    if net not in sizeOfNet:
        raise Exception("The net type error!")
    if not os.path.isdir(saveImagesFolder):
        os.makedirs(saveImagesFolder)
    saveF = open(join(saveFolder, "landmark.txt"), 'w')
    imageCnt = 0
    # image_path bbox landmark(5*2)
    for (imgPath, bbox, landmarkGt) in getBboxLandmarkFromTxt(srcTxt):
        F_imgs = []
        F_landmarks = []
        img = cv2.imread(imgPath)
        assert (img is not None)
        img_h, img_w, img_c = img.shape
        gt_box = np.array([bbox.left, bbox.top, bbox.right, bbox.bottom])
        f_face = img[bbox.top: bbox.bottom + 1, bbox.left: bbox.right + 1]
        f_face = cv2.resize(f_face, (sizeOfNet[net] * 2, sizeOfNet[net]))
        landmark = np.zeros((2, 2))
        # normalize
        for index, one in enumerate(landmarkGt):
            rv = ((one[0] - gt_box[0]) / (gt_box[2] - gt_box[0]), (one[1] - gt_box[1]) / (gt_box[3] - gt_box[1]))
            landmark[index] = rv
        F_imgs.append(f_face)
        F_landmarks.append(landmark.reshape(4))
        landmark = np.zeros((2, 2))
        if augment:
            x1, y1, x2, y2 = gt_box
            # gt's width
            gt_w = x2 - x1 + 1
            # gt's height
            gt_h = y2 - y1 + 1
            if max(gt_w, gt_h) < 40 or x1 < 0 or y1 < 0:
                continue
            # random shift
            for i in range(40):
                # bbox_size = np.random.randint(int(min(gt_w, gt_h) * 0.8), np.ceil(1.25 * max(gt_w, gt_h)))

                size_w = np.random.randint(int(gt_w * 0.8), np.ceil(1.25 * gt_w))
                size_h = np.random.randint(int(gt_h * 0.8), np.ceil(1.25 * gt_h))

                delta_x = np.random.randint(-gt_w * 0.2, gt_w * 0.2)
                delta_y = np.random.randint(-gt_h * 0.2, gt_h * 0.2)
                nx1 = max(x1 + gt_w / 2 - size_w / 2 + delta_x, 0)
                ny1 = max(y1 + gt_h / 2 - size_h / 2 + delta_y, 0)

                nx2 = nx1 + size_w
                ny2 = ny1 + size_h
                if nx2 > img_w or ny2 > img_h:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])
                cropped_im = img[int(ny1):int(ny2 + 1), int(nx1):int(nx2 + 1), :]
                resized_im = cv2.resize(cropped_im, (sizeOfNet[net] * 2, sizeOfNet[net]))
                # cal iou
                iou = IoU(crop_box, np.expand_dims(gt_box, 0))
                if iou <= 0.8:
                    continue
                F_imgs.append(resized_im)
                # normalize
                for index, one in enumerate(landmarkGt):
                    rv = ((one[0] - nx1) / size_w, (one[1] - ny1) / size_h)
                    landmark[index] = rv
                F_landmarks.append(landmark.reshape(4))
                landmark = np.zeros((2, 2))
                landmark_ = F_landmarks[-1].reshape(-1, 2)
                bbox = BBox([nx1, ny1, nx2, ny2])

                # mirror
                if random.choice([0, 1]) > 0:
                    face_flipped, landmark_flipped = flip(resized_im, landmark_)
                    face_flipped = cv2.resize(face_flipped, (sizeOfNet[net] * 2, sizeOfNet[net]))
                    # c*h*w
                    F_imgs.append(face_flipped)
                    F_landmarks.append(landmark_flipped.reshape(4))
                # rotate
                if random.choice([0, 1]) > 0:
                    face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                     bbox.reprojectLandmark(landmark_), 5)  # 逆时针旋转
                    # landmark_offset
                    landmark_rotated = bbox.projectLandmark(landmark_rotated)
                    face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (sizeOfNet[net] * 2, sizeOfNet[net]))
                    F_imgs.append(face_rotated_by_alpha)
                    F_landmarks.append(landmark_rotated.reshape(4))

                    # flip
                    face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                    face_flipped = cv2.resize(face_flipped, (sizeOfNet[net] * 2, sizeOfNet[net]))
                    F_imgs.append(face_flipped)
                    F_landmarks.append(landmark_flipped.reshape(4))

                # inverse clockwise rotation
                if random.choice([0, 1]) > 0:
                    face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                     bbox.reprojectLandmark(landmark_), -5)  # 顺时针旋转
                    landmark_rotated = bbox.projectLandmark(landmark_rotated)
                    face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (sizeOfNet[net] * 2, sizeOfNet[net]))
                    F_imgs.append(face_rotated_by_alpha)
                    F_landmarks.append(landmark_rotated.reshape(4))

                    face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                    face_flipped = cv2.resize(face_flipped, (sizeOfNet[net] * 2, sizeOfNet[net]))
                    F_imgs.append(face_flipped)
                    F_landmarks.append(landmark_flipped.reshape(4))
        F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)
        for i in range(len(F_imgs)):
            path = os.path.join(saveImagesFolder, "%d.jpg" % (imageCnt))
            cv2.imwrite(path, F_imgs[i])
            landmarks = map(str, list(F_landmarks[i]))
            saveF.write(path + " -2 " + " ".join(landmarks) + "\n")
            imageCnt += 1
        printStr = "\rCount: {}".format(imageCnt)
        sys.stdout.write(printStr)
        sys.stdout.flush()
    saveF.close()
    print("\nLandmark create done!")


if __name__ == "__main__":
    net = "net2_v2"
    gen_landmark_data("images1/gt.txt", net, augment=True)

