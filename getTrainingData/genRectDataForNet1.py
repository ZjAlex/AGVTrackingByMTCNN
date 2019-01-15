#coding:utf-8
import sys
import numpy as np
import cv2
import os
rootPath = "D:/alex/CascadeNetwork3/"
from toolsFunc.commomUtils import IoU

neg_iou_threshhold = 0.3
part_iou_threshhold = 0.4
pos_iou_threshhold = 0.6
def gen_hard_bbox_pnet(srcAnnotations):
    srcAnnotations = os.path.join(rootPath, srcAnnotations)
    saveFolder = os.path.join(rootPath, "tmp/data/net1/")
    print(">>>>>> Gen hard samples for net1...")
    typeName = ["pos", "neg", "part"]
    saveFiles = {}
    for tp in typeName:
        _saveFolder = os.path.join(saveFolder, tp)
        if not os.path.isdir(_saveFolder):
            os.makedirs(_saveFolder)
        saveFiles[tp] = open(os.path.join(saveFolder, "{}.txt".format(tp)), 'w')

    annotationsFile = open(srcAnnotations, "r")
    pIdx = 0 # positive
    nIdx = 0 # negative
    dIdx = 0 # dont care
    idx = 0
    for annotation in annotationsFile:
        annotation = annotation.strip().split(' ')
        # image path
        imPath = annotation[0]
        # boxed change to float typ
        x1_ = float(annotation[1])
        y1_ = float(annotation[2])
        x2_ = float(annotation[1]) + float(annotation[3])
        y2_ = float(annotation[2]) + float(annotation[4])
        # gt. each row mean bounding box
        bbox = [x1_, y1_, x2_, y2_]
        # gt. each row mean bounding box
        boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
        #load image
        img = cv2.imread(os.path.join(rootPath, imPath))
        idx += 1
        height, width, channel = img.shape

        # 1. NEG: random to crop negative sample image
        negNum = 0
        while negNum < 20:
            size_y = np.random.randint(12, width / 4)
            size_x = size_y * 2
            if size_x > height:
                continue

            # top_left
            nx = np.random.randint(0, width - size_x)
            ny = np.random.randint(0, height - size_y)
            # random crop
            crop_box = np.array([nx, ny, nx + size_x, ny + size_y])
            # cal iou and iou must below 0.3 for neg sample
            iou = IoU(crop_box, boxes)
            if np.max(iou) >= neg_iou_threshhold:
                continue
            # crop sample image
            cropped_im = img[ny : ny + size_y, nx : nx + size_x, :]
            resized_im = cv2.resize(cropped_im, (24, 12), interpolation=cv2.INTER_LINEAR)
            # now to save it
            save_file = os.path.join(saveFolder, "neg", "%s.jpg"%nIdx)
            saveFiles['neg'].write(save_file + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            nIdx += 1
            negNum += 1
        for box in boxes:
            x1, y1, x2, y2 = box
            #bbox's width and height
            w, h = x2 - x1 + 1, y2 - y1 + 1
            # ignore small objects
            if max(w, h) < 40 or x1 < 0 or y1 < 0:
                continue
            # 2. NEG: random to crop sample image in bbox inside
            for i in range(5):
                size_y = np.random.randint(12, width / 4)
                size_x = size_y * 2
                if size_x > height:
                    continue

                # delta_x and delta_y are offsets of (x1, y1)
                delta_x = np.random.randint(max(-size_x, -x1), w)
                delta_y = np.random.randint(max(-size_y, -y1), h)
                nx1 = int(max(0, x1 + delta_x))
                ny1 = int(max(0, y1 + delta_y))
                if nx1 + size_x > width or ny1 + size_y > height:
                    continue
                crop_box = np.array([nx1, ny1, nx1 + size_x, ny1 + size_y])
                Iou = IoU(crop_box, boxes)
                if np.max(Iou) >= neg_iou_threshhold:
                    continue
                cropped_im = img[ny1: ny1 + size_y, nx1: nx1 + size_x, :]
                resized_im = cv2.resize(cropped_im, (24, 12), interpolation=cv2.INTER_LINEAR)
                save_file = os.path.join(saveFolder, "neg", "%s.jpg"%nIdx)
                saveFiles['neg'].write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                nIdx += 1
            # 3. POS and PART
            for i in range(20):
                # pos and part face size [minsize*0.8,maxsize*1.25]
                # size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
                size_y = np.random.randint(int(h * 0.8), int(h * 1.2))
                size_x = size_y * 2
                # delta here is the offset of box center
                delta_x = np.random.randint(int(-w * 0.2), w * 0.2)
                delta_y = np.random.randint(int(-h * 0.2), h * 0.2)
                #show this way: nx1 = max(x1+w/2-size/2+delta_x)
                nx1 = max(x1 + w / 2 + delta_x - size_x / 2, 0)
                #show this way: ny1 = max(y1+h/2-size/2+delta_y)
                ny1 = max(y1 + h / 2 + delta_y - size_y / 2, 0)
                nx2 = nx1 + size_x
                ny2 = ny1 + size_y

                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])
                #yu gt de offset
                offset_x1 = (x1 - nx1) / float(size_x)
                offset_y1 = (y1 - ny1) / float(size_y)
                offset_x2 = (x2 - nx2) / float(size_x)
                offset_y2 = (y2 - ny2) / float(size_y)
                #crop
                cropped_im = img[int(ny1) : int(ny2), int(nx1) : int(nx2), :]
                #resize
                resized_im = cv2.resize(cropped_im, (24, 12), interpolation=cv2.INTER_LINEAR)

                box_ = box.reshape(1, -1)
                if IoU(crop_box, box_) >= pos_iou_threshhold:
                    save_file = os.path.join(saveFolder, "pos", "%s.jpg"%pIdx)
                    saveFiles['pos'].write(save_file + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    pIdx += 1
                elif IoU(crop_box, box_) >= part_iou_threshhold:
                    save_file = os.path.join(saveFolder, "part", "%s.jpg"%dIdx)
                    saveFiles['part'].write(save_file + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    dIdx += 1
        printStr = "\r[{}] pos: {}  neg: {}  part:{}".format(idx, pIdx, nIdx, dIdx)
        sys.stdout.write(printStr)
        sys.stdout.flush()
    for f in saveFiles.values():
        f.close()
    print('\n')


if __name__ == "__main__":
    gen_hard_bbox_pnet( "images1/gt.txt")
