import cv2
import time
import numpy as np
import sys
import os
from detectors.nms import py_nms
from models.configs import config
import time

rootPath = "D:/alex/CascadeNetwork3/"


class WholeDetector(object):
    def __init__(self,
                 detectors,
                 min_face_size=48,
                 stride=2,
                 threshold=[0.6, 0.7, 0.7],
                 scale_factor=0.7):
        self.net1_detector = detectors[0]
        self.net2_detector = detectors[1]
        self.net3_detector = detectors[2]
        self.min_face_size = min_face_size
        self.stride = stride
        self.thresh = threshold
        self.scale_factor = scale_factor

    def convert_to_square(self, bbox):
        """
            convert bbox to square
        Parameters:
        ----------
            bbox: numpy array , shape n x 5
                input bbox
        Returns:
        -------
            square bbox
        """
        square_bbox = bbox.copy()

        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        max_side = np.maximum(h, w)
        square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
        square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
        square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
        square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
        return square_bbox

    def calibrate_box(self, bbox, reg):
        """
            calibrate bboxes
        Parameters:
        ----------
            bbox: numpy array, shape n x 5
                input bboxes
            reg:  numpy array, shape n x 4
                bboxes adjustment
        Returns:
        -------
            bboxes after refinement
        """

        bbox_c = bbox.copy()
        w = bbox[:, 2] - bbox[:, 0] + 1
        w = np.expand_dims(w, 1)
        h = bbox[:, 3] - bbox[:, 1] + 1
        h = np.expand_dims(h, 1)
        reg_m = np.hstack([w, h, w, h])
        aug = reg_m * reg
        bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
        return bbox_c

    def generate_bbox(self, cls_map, reg, scale, threshold):
        """
            generate bbox from feature cls_map
        Parameters:
        ----------
            cls_map: numpy array , n x m
                detect score for each position
            reg: numpy array , n x m x 4
                bbox
            scale: float number
                scale of this detection
            threshold: float number
                detect threshold
        Returns:
        -------
            bbox array
        """
        cellsize_y = 12
        cellsize_x = 24
        t_index = np.where(cls_map > threshold)
        # find nothing
        if t_index[0].size == 0:
            return np.array([])
        # offset
        dx1, dy1, dx2, dy2 = [reg[t_index[0], t_index[1], i] for i in range(4)]
        reg = np.array([dx1, dy1, dx2, dy2])
        score = cls_map[t_index[0], t_index[1]]
        boundingbox = np.vstack([np.round((self.stride * t_index[1]) / scale),
                                 np.round((self.stride * t_index[0]) / scale),
                                 np.round((self.stride * t_index[1] + cellsize_x) / scale),
                                 np.round((self.stride * t_index[0] + cellsize_y) / scale),
                                 score,
                                 reg])
        return boundingbox.T

    def processed_image(self, img, scale):
        height, width, channels = img.shape
        new_height = int(height * scale)  # resized new height
        new_width = int(width * scale)  # resized new width
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image
        img_resized = (img_resized - 127.5) / 128
        return img_resized

    def pad(self, bboxes, w, h):
        """
            pad the the bboxes, alse restrict the size of it
        Parameters:
        ----------
            bboxes: numpy array, n x 5
                input bboxes
            w: float number
                width of the input image
            h: float number
                height of the input image
        Returns :
        ------
            dy, dx : numpy array, n x 1
                start point of the bbox in target image
            edy, edx : numpy array, n x 1
                end point of the bbox in target image
            y, x : numpy array, n x 1
                start point of the bbox in original image
            ex, ex : numpy array, n x 1
                end point of the bbox in original image
            tmph, tmpw: numpy array, n x 1
                height and width of the bbox
        """
        tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
        num_box = bboxes.shape[0]
        dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
        edx, edy = tmpw.copy() - 1, tmph.copy() - 1
        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        tmp_index = np.where(ex > w - 1)
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
        ex[tmp_index] = w - 1
        tmp_index = np.where(ey > h - 1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1
        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0
        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0
        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]
        return return_list

    def get_net1_feat_map(self, im):
        h, w, c = im.shape
        net_size_y = 12
        net_size_x = 24

        current_scale = float(net_size_x) / self.min_face_size  # find initial scale
        im_resized = self.processed_image(im, current_scale)
        current_height, current_width, _ = im_resized.shape
        # for fcn
        all_boxes = list()

        index = 0
        while current_height > net_size_y and current_width > net_size_x:
            # return the result predicted by pnet
            # cls_cls_map : H*w*2
            # reg: H*w*4
            index += 1
            cls_cls_map, reg, net, conv1_, conv_2 = self.net1_detector.predict(im_resized)
            print(net.shape)
            for i in range(net.shape[3]):
                cv2.imshow(str(index), net[0, :, :, i:i+1])
                if cv2.waitKey(0) == 27:
                    continue
            current_scale *= self.scale_factor
            im_resized = self.processed_image(im, current_scale)
            current_height, current_width, _ = im_resized.shape

    def detect_net1(self, im):
        """Get face candidates through pnet

        Parameters:
        ----------
        im: numpy array
            input image array

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        h, w, c = im.shape
        net_size_y = 12
        net_size_x = 24

        current_scale = float(net_size_x) / self.min_face_size  # find initial scale
        im_resized = self.processed_image(im, current_scale)
        current_height, current_width, _ = im_resized.shape
        # for fcn
        all_boxes = list()
        while current_height > net_size_y and current_width > net_size_x:
            # return the result predicted by pnet
            # cls_cls_map : H*w*2
            # reg: H*w*4
            cls_cls_map, reg = self.net1_detector.predict(im_resized)
            # boxes: num*9(x1,y1,x2,y2,score,x1_offset,y1_offset,x2_offset,y2_offset)
            boxes = self.generate_bbox(cls_cls_map[:, :, 1], reg, current_scale, self.thresh[0])

            current_scale *= self.scale_factor
            im_resized = self.processed_image(im, current_scale)
            current_height, current_width, _ = im_resized.shape

            if boxes.size == 0:
                continue
            keep = py_nms(boxes[:, :5], config.net1_in_nms_iou_threshold, 'Union')
            boxes = boxes[keep]
            all_boxes.append(boxes)
        if len(all_boxes) == 0:
            return None, None, None
        all_boxes = np.vstack(all_boxes)
        # merge the detection from first stage
        keep = py_nms(all_boxes[:, 0:5], config.net1_out_nms_iou_threshold, 'Union')
        all_boxes = all_boxes[keep]
        boxes = all_boxes[:, :5]
        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
        # refine the boxes
        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])
        boxes_c = boxes_c.T
        return boxes, boxes_c, None

    def detect_net2(self, im, dets):
        """Get face candidates using rnet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of pnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        h, w, c = im.shape
        # dets = self.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 24, 48, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (48, 24)) - 127.5) / 128
        # cls_scores : num_data*2
        # reg: num_data*4
        # landmark: num_data*10
        cls_scores, reg= self.net2_detector.predict(cropped_ims)
        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > self.thresh[1])[0]
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
        else:
            return None, None, None
        keep = py_nms(boxes, config.net2_nms_iou_threshold)
        boxes = boxes[keep]
        boxes_c = self.calibrate_box(boxes, reg[keep])
        return boxes, boxes_c, None

    def detect_net2_v2(self, im, dets):
        """Get face candidates using rnet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of pnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        h, w, c = im.shape
        # dets = self.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 24, 48, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (48, 24)) - 127.5) / 128
        cls_scores,reg, landmark = self.net2_detector.predict(cropped_ims)
        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > self.thresh[2])[0]
        if len(keep_inds) > 0:
            # pickout filtered box
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None, None
        # pickout filtered box

        # width
        w = boxes[:, 2] - boxes[:, 0] + 1
        # height
        h = boxes[:, 3] - boxes[:, 1] + 1
        landmark[:, 0::2] = (np.tile(w, (2, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (2, 1)) - 1).T
        landmark[:, 1::2] = (np.tile(h, (2, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (2, 1)) - 1).T
        boxes_c = self.calibrate_box(boxes, reg)

        keep = py_nms(boxes_c, config.net2_nms_iou_threshold, "Minimum")
        boxes = boxes[keep]
        boxes_c = boxes_c[keep]
        landmark = landmark[keep]
        # boxes = boxes[py_nms(boxes, config.net3_nms_iou_threshold, "Minimum")]
        # keep = py_nms(boxes_c, config.net3_nms_iou_threshold, "Minimum")
        # boxes_c = boxes_c[keep]
        # landmark = landmark[keep]
        return boxes, boxes_c, landmark

    def detect_net3(self, im, dets):
        """Get face candidates using onet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of rnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        h, w, c = im.shape
        # dets = self.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 48, 96, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (96, 48)) - 127.5) / 128
        reg, landmark = self.net3_detector.predict(cropped_ims)

        # pickout filtered box
        boxes = dets
        boxes[:, 4] = 1.0
        reg = reg
        landmark = landmark

        # width
        w = boxes[:, 2] - boxes[:, 0] + 1
        # height
        h = boxes[:, 3] - boxes[:, 1] + 1
        landmark[:, 0::2] = (np.tile(w, (2, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (2, 1)) - 1).T
        landmark[:, 1::2] = (np.tile(h, (2, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (2, 1)) - 1).T
        boxes_c = self.calibrate_box(boxes, reg)
       # boxes = boxes[py_nms(boxes, config.net3_nms_iou_threshold, "Minimum")]
       # keep = py_nms(boxes_c, config.net3_nms_iou_threshold, "Minimum")
       # boxes_c = boxes_c[keep]
       # landmark = landmark[keep]
        return boxes, boxes_c, landmark

    def get_feats_from_net3(self, im, dets):
        """Get face candidates using onet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of rnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        h, w, c = im.shape
        # dets = self.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 48, 96, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (96, 48)) - 127.5) / 128
        reg, landmark = self.net3_detector.predict(cropped_ims)
        print(net.shape)
        for i in range(net.shape[3]):
            cv2.imshow("feats", net[0, :, :, i:i + 1])
            if cv2.waitKey(0) == 27:
                continue

        # pickout filtered box
        boxes = dets
        boxes[:, 4] = 1.0
        reg = reg
        landmark = landmark

        # width
        w = boxes[:, 2] - boxes[:, 0] + 1
        # height
        h = boxes[:, 3] - boxes[:, 1] + 1
        landmark[:, 0::2] = (np.tile(w, (2, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (2, 1)) - 1).T
        landmark[:, 1::2] = (np.tile(h, (2, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (2, 1)) - 1).T
        boxes_c = self.calibrate_box(boxes, reg)
       # boxes = boxes[py_nms(boxes, config.net3_nms_iou_threshold, "Minimum")]
       # keep = py_nms(boxes_c, config.net3_nms_iou_threshold, "Minimum")
       # boxes_c = boxes_c[keep]
       # landmark = landmark[keep]
        return boxes, boxes_c, landmark

    def detect_face(self, test_data):
        all_boxes = []  # save each image's bboxes
        landmarks = []
        batch_idx = 0
        for databatch in test_data:
            # print info
            printStr = "\rDone images: {}".format(batch_idx)
            sys.stdout.write(printStr)
            sys.stdout.flush()
            batch_idx += 1
            im = databatch
            # net1
            start1 = time.time()
            if self.net1_detector:
                # ignore landmark
                boxes, boxes_c, landmark = self.detect_net1(im)
                if boxes_c is None:
                    all_boxes.append(np.array([]))
                    landmarks.append(np.array([]))
                    continue
            # net2
            end1 = time.time()
            inf1 = end1 - start1
         #   print('\n')
           # print("inference1 time: " + str(end1 - start1))
            start2 = time.time()
            if self.net2_detector:
                # ignore landmark
                boxes, boxes_c, landmark = self.detect_net2(im, boxes_c)
                if boxes_c is None:
                    all_boxes.append(np.array([]))
                    landmarks.append(np.array([]))
                    continue
            # net3
            end2 = time.time()
            inf2 = end2 - start2
           # print("inference2 time: " + str(end2 - start2))

            start3 = time.time()
            if self.net3_detector:
                boxes, boxes_c, landmark = self.detect_net3(im, boxes_c)
                if boxes_c is None:
                    all_boxes.append(np.array([]))
                    landmarks.append(np.array([]))
                    continue
            end3 = time.time()
            inf3 = end3 - start3
           # print("inference3 time: " + str(end3 - start3))

           # print("inference total time: " + str(end3 - start3 + end2 - start2 + end1 - start1))

            all_boxes.append(boxes_c)
            landmarks.append(landmark)
        return all_boxes, landmarks


    def get_feat_map(self, im):
        boxes, boxes_c, landmark = self.detect_net1(im)
        return

    def detect_face_oneImage(self, im):
        all_boxes = []  # save each image's bboxes
        landmarks = []
        inf1 = 0
        inf2 = 0
        inf3 = 0
        pred_prob = 0
        # net1
        start1 = time.time()
        if self.net1_detector:
            # ignore landmark
            boxes, boxes_c, landmark = self.detect_net1(im)
            if boxes_c is None:
                all_boxes.append(np.array([]))
                landmarks.append(np.array([]))
                return all_boxes, landmarks, inf1, inf2, inf3,pred_prob
        # net2
        end1 = time.time()
        inf1 = end1 - start1
        start2 = time.time()
        if self.net2_detector:
            # ignore landmark
            boxes, boxes_c, landmark = self.detect_net2(im, boxes_c)
            if boxes_c is None:
                all_boxes.append(np.array([]))
                landmarks.append(np.array([]))
                return all_boxes, landmarks, inf1, inf2, inf3,pred_prob
            pred_prob = boxes[0, 4]
        # net3
        end2 = time.time()
        inf2 = end2 - start2

        start3 = time.time()
        if self.net3_detector:
            boxes, boxes_c, landmark = self.detect_net3(im, boxes_c)
            if boxes_c is None:
                all_boxes.append(np.array([]))
                landmarks.append(np.array([]))
                return all_boxes, landmarks, inf1, inf2, inf3,pred_prob
        end3 = time.time()
        inf3 = end3 - start3


        all_boxes.append(boxes_c)
        landmarks.append(landmark)
        return all_boxes, landmarks, inf1, inf2, inf3, pred_prob


