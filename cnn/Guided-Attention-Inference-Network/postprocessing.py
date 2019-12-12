import os
import argparse
import chainer
import cv2
from chainer import Variable
import chainer.functions as F
import cupy as cp

from chainercv.datasets import VOCSemanticSegmentationDataset
from chainer.iterators import SerialIterator
from chainer.serializers import load_npz
from chainer.backends.cuda import get_array_module

from chainercv.utils import read_image

from matplotlib import pyplot as plt
import numpy as np

HAND_CLASS_ID = 20
THRESH_OBJ = 0.01
THRESH_HAND = 0.001

class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
        'hand/finger'
    ])
# class_colors: id: bgr scaling from gray
# chair: bottle: green, chair:red, table: yellow, person: cyan, sofa: purple, hand: blue
class_colors = {5:[0,1,0], 9: [0,0,1], 11: [0, 1, 1], 15:[1,1,0], 18:[1,0,1], 21: [1,0,0]}

# converts rectangles returned by boundingRect to [left, top, right, bottom]
def _contour_rect_to_corner_rect(rect):
    return [rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]]

def _area_of(corner_rect):
    return (corner_rect[2] - corner_rect[0]) * (corner_rect[3] - corner_rect[1])

def _distance(corner_rect, point):
    # if point is inside bounding rect, use distance from point to edges.
    # This gives smaller bounding rects less inner distance.

    # if corner_rect[0] <= point[0] <= corner_rect[2] and corner_rect[1] <= point[1] <= corner_rect[3]:
    #     dx = abs(max(corner_rect[0] - point[0], point[0] - corner_rect[2]))  # point inside the rect
    #     dy = abs(max(corner_rect[1] - point[1], point[1] - corner_rect[3]))
    # else:
    #     dx = max(corner_rect[0] - point[0], 0, point[0] - corner_rect[2])
    #     dy = max(corner_rect[1] - point[1], 0, point[1] - corner_rect[3])
    dx = max(corner_rect[0] - point[0], 0, point[0] - corner_rect[2])
    dy = max(corner_rect[1] - point[1], 0, point[1] - corner_rect[3])
    # print(dx)
    # print(dy)
    return int((dx*dx + dy*dy)**0.5)


def _class_id_to_str(class_id):
    if class_id is None:
        return "[None]"
    else:
        return class_names[int(class_id) + 1]

def _gcam_to_mask(gcam_np, class_id):
    class_id += 1 # ignore background class
    # print(class_id)
    thresh = THRESH_OBJ
    if class_id == 21:
        thresh = THRESH_HAND
    if class_id == 21:
        gcam_np[gcam_np < np.max(gcam_np) / 3] = 0
    else:
        gcam_np[gcam_np < np.max(gcam_np) / 8] = 0
    # gcam_np[gcam_np > thresh] = 1
    gcam_np = cv2.cvtColor(gcam_np, cv2.COLOR_GRAY2BGR)
    gcam_np *= (1/1.4/np.max(gcam_np))
    gcam_np[...] = np.uint32(gcam_np[...] * 255) # max 2^8 to prevent integer overflowing
    # if is_hand:
    #     gcam_np[..., (1,2)] = 0 # make mask green for hands
    if class_id in class_colors.keys():
        # print(class_colors[class_id])
        gcam_np[..., :] = np.uint32(np.multiply(gcam_np[..., :], np.array(class_colors[class_id])))
    else:
        pass # keep mask color gray if not in object categories we care about
    return gcam_np

def gcams_to_mask(gcams_from_chainer, class_ids, dataset=None, img=None):
    if len(class_ids) == 0:
        return None
    gcams_np = []
    gcam_aggregate = None
    for i in range(len(gcams_from_chainer)):
        # gcam for class i
        gcams_np.append(cp.asnumpy(F.squeeze(gcams_from_chainer[i][0], 0).data))
    print(class_ids)

    for i in range(len(gcams_np)):
        # so earlier indices will have brighter heatmaps
        gcam_np = gcams_np[i]
        print("Max gcam magnitude for {}: ".format(class_names[int(class_ids[i]) + 1]) + str(np.max(gcams_np[i])))
        print("Min gcam magnitude for {}: ".format(class_names[int(class_ids[i]) + 1]) + str(np.min(gcams_np[i])))
        mask = _gcam_to_mask(gcam_np, int(class_ids[i]))
        assert mask != None
        cv2.imshow("mask", np.uint8(mask))
        cv2.waitKey(0)

        if gcam_aggregate is None:
            gcam_aggregate = mask
        else:
            gcam_aggregate = gcam_aggregate + mask
    return gcam_aggregate

# returns top left, top right, width, height
def contours_to_bboxes(contours, class_id):
    if len(contours) == 0:
        return None
    if class_id == 20: # special case for hand, where we assume there can be at most one hand.
        c = max(contours, key=cv2.contourArea)
        # determine the most extreme points along the contour
        # extLeft = tuple(c[c[:, :, 0].argmin()][0])
        # extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        print(extTop)
        return [_contour_rect_to_corner_rect([extTop[0], extTop[1], 10, 10])]
        # extBot = tuple(c[c[:, :, 1].argmax()][0])
        # maxArea = 0
        # boundRects = [None]
        # for i, c in enumerate(contours):
        #     contours_poly = cv2.approxPolyDP(c, 3, True)
        #     boundRect = cv2.boundingRect(contours_poly)
        #     area = boundRect[2] * boundRect[3]
        #     if area > maxArea:
        #         maxArea = area
        #         boundRects[0] = boundRect  # there can only be one best match (max area) rectangle for hand
    else:
        contours_poly = []
        boundRects = []
        for i, c in enumerate(contours):
            if cv2.contourArea(c) < 20*20:
                continue
            contours_poly.append(cv2.approxPolyDP(c, 3, True))
            boundRects.append(_contour_rect_to_corner_rect(cv2.boundingRect(contours_poly[-1])))
    return boundRects

def _gcam_to_bboxes(gcam, class_id, thresh, thresh_hand):
    # print(class_id)
    threshold = thresh
    if class_id == 20:
        threshold = thresh_hand

    gcam[gcam < np.max(gcam) * threshold] = 0
    gcam[...] = gcam[...] * 255
    gcam = np.uint8(gcam)
    # gcam_np = cv2.cvtColor(gcam_np, cv2.COLOR_GRAY2BGR)
    contours, hierarchy = cv2.findContours(gcam.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bboxes = contours_to_bboxes(contours, class_id)
    out = np.zeros(gcam.shape)
    cv2.drawContours(out, contours, -1, 255, 2)
    for i in range(len(bboxes)):
        cv2.rectangle(out, (int(bboxes[i][0]), int(bboxes[i][1])), (bboxes[i][2], int(bboxes[i][3])), 255, 2)
    cv2.imshow('contours for {}'.format(_class_id_to_str(class_id)), out)

    return bboxes

def gcams_to_bboxes(gcams_from_chainer, class_ids, threshold=1/4, threshold_hand=1/5, input_image=None):
    """
    :param threshold: all gcam pixels less than threshold*np.max(gcam) will be ignored in bounding box calculation
    :param threshold_hand: a higher threshold used for the hand/finger (because we're really trying to find a point)
    , not an area
    :return: An array of bounding boxes in the order of the class ids (if a hand is present, its bounding box will be the last element
    in the return array). The bounding boxes are in corner form: [left, top, right, bottom]
    """
    if input_image is not None:
        cv2.imshow('input img', input_image)
    if len(class_ids) == 0:
        return None, None
    gcams_np = []
    bboxes_per_class = []
    for i in range(len(gcams_from_chainer)):
        # gcam for class i
        gcams_np.append(cp.asnumpy(F.squeeze(gcams_from_chainer[i][0], 0).data))
    print(class_ids)

    for i in range(len(gcams_np)):
        # so earlier indices will have brighter heatmaps
        gcam_np = gcams_np[i]
        print("Max gcam magnitude for {}: ".format(_class_id_to_str(class_ids[i])) + str(np.max(gcams_np[i])))
        print("Min gcam magnitude for {}: ".format(_class_id_to_str(class_ids[i])) + str(np.min(gcams_np[i])))
        bboxes = _gcam_to_bboxes(gcam_np, int(class_ids[i]), threshold, threshold_hand)
        bboxes_per_class.append(bboxes)

    min_distance = 100000 # the object being pointed at is the one with min distance from its bounding box to fingertip
    pointed_obj_class = None
    pointed_bbox = None
    if class_ids[-1] == 20:
        fingertip_loc = (bboxes_per_class[-1][0][0], bboxes_per_class[-1][0][1])
        for class_index, bboxes in enumerate(bboxes_per_class[:-1]):  # skip hand bounding boxes
            for bbox in bboxes:
                dist = _distance(bbox, fingertip_loc)
                if dist < min_distance:
                    min_distance = dist
                    pointed_obj_class = int(class_ids[class_index])
                    pointed_bbox = bbox
                elif dist == min_distance: # if two distances are the same, prioritize the smaller bounding box
                    if _area_of(bbox) < _area_of(pointed_bbox):
                        pointed_obj_class = int(class_ids[class_index])
                        pointed_bbox = bbox
                    else:
                        print("ignoring bigger bbox from {}".format(_class_id_to_str(class_ids[class_index])))
        print("Pointed Object Class: {} with distance {}, bbox is {}.".format(_class_id_to_str(pointed_obj_class), min_distance, pointed_bbox))
    cv2.waitKey(0)

    return bboxes_per_class, pointed_bbox