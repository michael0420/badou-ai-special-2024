import numpy as np
import tensorflow as tf
from PIL import Image

class BBoxUtility(object):
    def __init__(self, priors=None, overlap_threshold=0.7, ignore_threshold=0.3,
                 nms_thresh=0.7, top_k=400):
        self.overlap_threshold = overlap_threshold
        self.priors = priors
        self.num_priors = 0 if priors is None else len(priors)
        self.ignore_threshold = ignore_threshold
        self._nms_thresh = nms_thresh # Non-maximum suppression threshold
        self._top_k = top_k
        self.boxes = tf.placeholder(dtype='float32', shape=(None, 4))
        self.scores = tf.placeholder(dtype='float32', shape=(None,))
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores, 
                                                self.top_k, 
                                                iou_threshold=self.nms_thresh)
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

    @property
    def nms_thresh(self):
        return self._nms_thresh

    @nms_thresh.setter
    def nms_thresh(self, value):
        self._nms_thresh = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores, 
                                                self._top_k, 
                                                iou_threshold=self._nms_thresh)
    
    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, value):
        self._top_k = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores, 
                                                self._top_k, 
                                                iou_threshold=self._nms_thresh)
    def iou(self, box):
        """Compute intersection over union for the box with all priors.
        """
        inter_upleft = np.maximum(self.priors[:, :2], box[:2])
        inter_botright = np.minimum(self.priors[:, 2:4], box[2:])

        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # calculate the area of the box true
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        # calculate the area of the priors
        area_gt = area_gt = (self.priors[:, 2] - self.priors[:, 0])*(self.priors[:, 3] - self.priors[:, 1])

        union = area_true + area_gt - inter

        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True):
        iou = self.iou(box)
        encoded_box = np.zeros((self.num_priors, 4 + return_iou))

        # assgin the max iou prior to the box
        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        
        # get the assigned priors
        assigned_priors = self.priors[assign_mask]
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]

        assigned_priors_center = 0.5 * (assigned_priors[:, :2] +
                                        assigned_priors[:, 2:4])
        assigned_priors_wh = (assigned_priors[:, 2:4] -
                              assigned_priors[:, :2])
        
        # calculate the encode box and normalize it
        # center offset
        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= assigned_priors_wh
        encoded_box[:, :2][assign_mask] *= 4

        # height and width scale
        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)
        encoded_box[:, 2:4][assign_mask] *= 4

        # flatten the encoded_box
        return encoded_box.ravel()

    def ignore_box(self, box):
        iou = self.iou(box)
        ignored_box = np.zeros((self.num_priors, 1))

        # find the the value of iou between ignore_threshold and overlap_threshold
        assign_mask = (iou > self.ignore_threshold)&(iou<self.overlap_threshold)
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
            
        ignored_box[:, 0][assign_mask] = iou[assign_mask]
        return ignored_box.ravel()

    def assign_boxes(self, boxes, anchors):
        self.num_priors = len(anchors)
        self.priors = anchors
        assignment = np.zeros((self.num_priors, 4 + 1))

        # if no boxes, return assignment
        assignment[:, 4] = 0.0
        if len(boxes) == 0:
            return assignment
            
        # calculate the iou between the boxes and the priors
        ingored_boxes = np.apply_along_axis(self.ignore_box, 1, boxes[:, :4])
        ingored_boxes = ingored_boxes.reshape(-1, self.num_priors, 1)
        ignore_iou = ingored_boxes[:, :, 0].max(axis=0)
        ignore_iou_mask = ignore_iou > 0

        assignment[:, 4][ignore_iou_mask] = -1 # set the ignore box to -1

        # calculate the encoded box
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)

        # choose the max iou prior to the box
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        # get the index of the best iou prior
        best_iou_idx = best_iou_idx[best_iou_mask]

        # assign the encoded boxes to the priors
        assign_num = len(best_iou_idx)
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]

        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx,np.arange(assign_num),:4]
        assignment[:, 4][best_iou_mask] = 1 # set the positive box to 1

        # return assignment, which include the encoded box and the label to each prior
        return assignment

    
    # this fucntion is decode the box bound from the perdiction of the model
    def decode_boxes(self, mbox_loc, mbox_priorbox):
        # get the prior box width and height
        prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
        prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]

        # get the prior box center
        prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
        prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])

        # get the decode box center
        decode_bbox_center_x = mbox_loc[:, 0] * prior_width / 4
        decode_bbox_center_x += prior_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * prior_height / 4
        decode_bbox_center_y += prior_center_y
        
        # get the decode box width and height
        decode_bbox_width = np.exp(mbox_loc[:, 2] / 4)
        decode_bbox_width *= prior_width
        decode_bbox_height = np.exp(mbox_loc[:, 3] /4)
        decode_bbox_height *= prior_height

        # get the decode box bound
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        # stack the decode box bound
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), axis=-1)
        # limit the bound in the range of 0 and 1
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox


    def detection_out(self, predictions, mbox_priorbox, num_classes, keep_top_k=300,
                        confidence_threshold=0.5):
        
        # get the confidence and location prediction
        mbox_conf = predictions[0]
        mbox_loc = predictions[1]

        results = []

        # process the confidence and location prediction of each image
        for i in range(len(mbox_loc)):
            results.append([])
            decode_bbox = self.decode_boxes(mbox_loc[i], mbox_priorbox)

            # process each class
            for c in range(num_classes):
                c_confs = mbox_conf[i, :, c]
                c_confs_m = c_confs > confidence_threshold
                if len(c_confs[c_confs_m]) > 0:
                    # get the box bound and confidence of the class
                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]
                    # do the non-maximum suppression and choose the best box
                    feed_dict = {self.boxes: boxes_to_process,
                                    self.scores: confs_to_process}
                    idx = self.sess.run(self.nms, feed_dict=feed_dict)
                    good_boxes = boxes_to_process[idx]
                    confs = confs_to_process[idx][:, None]

                    # stack the label, confidence and box bound
                    labels = c * np.ones((len(idx), 1))
                    c_pred = np.concatenate((labels, confs, good_boxes),
                                            axis=1)
                    results[-1].extend(c_pred)

            # if the current image has result, then sort the result using the confidence
            # and choose the top_k boxes
            if len(results[-1]) > 0:
                # sort
                results[-1] = np.array(results[-1])
                argsort = np.argsort(results[-1][:, 1])[::-1]
                results[-1] = results[-1][argsort]
                results[-1] = results[-1][:keep_top_k]

        # using prior box and the prediction of Retinanet, we can get the real box
        return results
    
    # do the non-maximum suppression for the output of the model
    # remove the redundant box and keep the best box
    def nms_for_out(self,all_labels,all_confs,all_bboxes,num_classes,nms):
        results = []
        nms_out = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=nms)
        # process each class
        for c in range(num_classes):
            c_pred = []
            mask = all_labels == c
            # if current calss has box
            if len(all_confs[mask]) > 0:
                # get the box bound and confidence of the class
                boxes_to_process = all_bboxes[mask]
                confs_to_process = all_confs[mask]
                # do the non-maximum suppression and choose the best box
                feed_dict = {self.boxes: boxes_to_process,
                                self.scores: confs_to_process}
                idx = self.sess.run(nms_out, feed_dict=feed_dict)

                good_boxes = boxes_to_process[idx]
                confs = confs_to_process[idx][:, None]
                # stack the label, confidence and box bound
                labels = c * np.ones((len(idx), 1))
                c_pred = np.concatenate((labels, confs, good_boxes),axis=1)
            results.extend(c_pred)
        return results
    