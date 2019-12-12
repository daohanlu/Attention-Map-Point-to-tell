from os import path, getcwd
import json
import random

import cv2
import numpy as np
from tqdm import tqdm

from utils.hand_dataset_loader import HandDatasetLoader
from utils.ade_hand_dataset_maker import AdeHandDatasetMaker
from utils.voc import SBDClassSeg

ADE_DATASET_ROOT = "semantic-segmentation-pytorch-master/data/"
HAND_DATASET_ROOT = "hands_data"

SAVE_METADATA_DIR = "VocHand_3"
SAVE_VERSION_NAME = "v3"
SAVE_DATASET_NAME = "VOCHand"

OBJ_IDS_OF_INTEREST = {5, 9, 11, 15, 18, 20}  # bottle chair table person sofa tv/monitor
OBJ_ID_HAND = 21

def overlap(r1, r2):
    '''Overlapping rectangles overlap both horizontally & vertically
    '''
    hoverlaps = True
    voverlaps = True
    if (r1[0] > r2[0] + r2[2]) or (r1[0] + r1[2] < r2[0]):
        hoverlaps = False
    if (r1[1] > r2[1] + r2[3]) or (r1[1] + r1[3] < r2[1]):
        voverlaps = False
    return hoverlaps and voverlaps


def intersection(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    return interArea


def area(r):
    # r: x, y, w, h
    return r[2] * r[3]


def parse_ade_input_list(odgt, max_sample=-1, start_idx=-1, end_idx=-1):
    """
    each entry in the return is formatted as the following:
    {'fpath_img': 'ADEChallengeData2016/images/training/ADE_train_00000001.jpg', 'fpath_segm': 'ADEChallengeData2016/annotations/training/ADE_train_00000001.png', 'width': 683, 'height': 512}
    :return: list of input files
    """
    list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]
    if max_sample > 0:
        list_sample = list_sample[0:max_sample]
    if start_idx >= 0 and end_idx >= 0:  # divide file list
        list_sample = list_sample[start_idx:end_idx]
    for entry in list_sample:
        entry['fpath_img'] = path.join(ADE_DATASET_ROOT, entry['fpath_img'])
        entry['fpath_segm'] = path.join(ADE_DATASET_ROOT, entry['fpath_segm'])
    num_sample = len(list_sample)
    assert num_sample > 0
    print('# ade samples: {}'.format(num_sample))
    return list_sample


def get_random_bboxes(segm_image, display_rgb=None, voc=False):
    """
    Gets a random set of bounding boxes from a segmentation image i,
    in with bboxes[i] = (x, y, w, h, obj_class)
    :return  a list of bounding boxes in which boxes[i] = (x, y, w, h, obj_class)
    """
    # interpret all non-background pixels as 255
    if not voc:
        segm_image = cv2.cvtColor(segm_image, cv2.COLOR_BGR2GRAY)
    uniques, counts = np.unique(segm_image, return_counts=True)
    obj_masks = []
    for obj_color in uniques:
        if obj_color == 0 or (voc and obj_color not in OBJ_IDS_OF_INTEREST):
            continue  # background color
        segm_copy = segm_image.copy()
        segm_copy[segm_copy != obj_color] = 0
        obj_masks.append(segm_copy)

    if display_rgb is not None:
        segm_image = display_rgb
    bounding_rects = []
    for obj_mask in obj_masks:
        obj_id = int(np.max(np.unique(obj_mask)))
        if obj_id not in OBJ_IDS_OF_INTEREST:
            continue
        contours, hierarchy = cv2.findContours(obj_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(0, len(contours)):
            # skip if size < 20*20 or contour is external (has a child within it)
            if cv2.contourArea(contours[i]) < 16 * 16:
                continue
            # bbox format: x, y, w, h, obj_id
            new_bbox = list(cv2.boundingRect(contours[i]))
            new_bbox.append(obj_id)
            add_new = True
            for old_bbox in bounding_rects:
                # keep the smaller box if new box contains or is contained in another
                if intersection(old_bbox, new_bbox) == min(area(old_bbox), area(new_bbox)):
                    if area(old_bbox) > area(new_bbox):
                        # x, y, w, h = old_bbox
                        # segm_image = cv2.rectangle(segm_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        old_bbox[:] = new_bbox[:]
                    else:
                        # x, y, w, h = new_bbox
                        # segm_image = cv2.rectangle(segm_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        pass
                    add_new = False
                    break
            if add_new:
                bounding_rects.append(new_bbox)
    bounding_rects = random.sample(bounding_rects, min(4, len(bounding_rects)))
    for rect in bounding_rects:
        x, y, w, h, obj_id = rect
        # segm_image = cv2.rectangle(segm_image, (x, y), (x + w, y + h), (obj_id, 0, 0), 2) # draw rect around each object
    return bounding_rects


def overlay_transparency(patch, target_image, startX, startY):
    """
    :param patch: Image to be overlayed in RGBA 4-channel representation
    """
    patch = np.copy(patch)
    max_patch_height = target_image.shape[0] - max(0, startY)
    max_patch_width = target_image.shape[1] - max(0, startX)
    # if patch is too tall (shouldn't happen in theory)
    target_width = target_image.shape[1] - startX
    if patch.shape[0] > max_patch_height:
        patch = patch[:max_patch_height, ...]
        print("g_transfer.py WARNING hand patch is too tall (should't happen)")

    # the fingertip is in the frame but left part of the hand is not
    if startX < 0:
        offset = - startX
        # max_patch_width -= startX + 1
        patch = patch[:, offset:]
    # the fingertip is in the frame but right part of the hand is not
    if startX > target_image.shape[1]:
        offset = startX - target_image.shape[1]
        patch = patch[:, :-offset]
        raise Exception("Error: Upper-left corner of hand image is to the right of fingertip location!")
    patch = patch[:, :max_patch_width]  # crop right part of the hand that's out of target image's right bound

    # startX = max(0, startX)
    # startY = max(0, startY)
    # target_area = target_image[startY: min(target_image.shape[0] - 1, startY + patch.shape[0]),
    #               startX: min(target_image.shape[1] - 1, startX + patch.shape[1]), ...]
    # print('target size' + str(target_area.shape))
    # # if patch is too large (pasting out of bounding), crop the patch image (hand)
    # if target_area.shape[0] != patch.shape[0] or target_area.shape[1] != patch.shape[1]:
    #     patch = patch[:target_area.shape[0], :target_area.shape[1], ...]
    #     print("hand overlay too large, cropping ...")
    #     # patch = patch[0:target_area.shape[0], 0:target_area.shape[1]]
    # print("Target upper-left point: " + str((startX, startY)))
    startX = max(startX, 0)  # since we've cropped the patches correctly, we can set negative indicies to 0
    startY = max(startY, 0)
    target_area = target_image[startY: startY + patch.shape[0],
                  startX: startX + patch.shape[1]]
    mask = patch[..., 3]
    # print("Target upper-left point after crop: " + str((startX, startY)))
    # print("patch size: " + str(patch.shape))
    # print("target area size: " + str(target_area.shape))
    target_area[mask == 255] = patch[..., [0, 1, 2]][mask == 255]
    target_image[startY:startY + patch.shape[0], startX: startX + patch.shape[1]] = target_area


def paste_hand_onto_img(original_img, bbox, hand_dataset):
    img_h = original_img.shape[0]
    img_w = original_img.shape[1]
    target_x = int(bbox[0] + bbox[2] * 0.5)  # fingertip would appear at right 50% and bottom 72% point from box origin
    target_y = int(bbox[1] + bbox[3] * 0.72)
    # cv2.rectangle(original_img, (target_x - 4, target_y - 4), (target_x + 4, target_y + 4), color=0x00FF00, thickness=2) # draw rect around fingertip
    # print("target {} in image {} width".format(target_x, img_w))
    hand_metadata = hand_dataset.get_random_hand_by_location(target_x / img_w, target_y / img_h)
    # print("hand_metadata: " + str(hand_metadata))
    fingertip_loc = hand_dataset.get_fingertip_loc_by_metadata(hand_metadata)
    hand_img = hand_dataset.get_image_by_metadata(hand_metadata)
    hand_new_height = img_h - target_y  # resize hand so that its height extends to bottom of the frame
    hand_resize_ratio = hand_new_height / hand_img.shape[0]
    hand_img = cv2.resize(hand_img, None, fx=hand_resize_ratio, fy=hand_resize_ratio)
    # original_img[target_y:target_y + hand_img.shape[0], target_x: target_x + hand_img.shape[1]] = hand_img
    actual_left = target_x - int(fingertip_loc[0] * hand_resize_ratio)
    actual_top = target_y - int(fingertip_loc[1] * hand_resize_ratio)
    overlay_transparency(hand_img, original_img, actual_left, actual_top)
    return [original_img], [hand_img], [(actual_left, actual_top)], [(target_x, target_y)]


def paste_hand_onto_mask(hand_image, original_mask, actual_top_left):
    """
    Paste the mask of the hand onto a mask image, given the top-left target point of the hand.
    """
    mask_hand = hand_image.copy()
    # color the hand as OBJ_ID_HAND
    mask_hand[mask_hand[..., 3] != 0] = [OBJ_ID_HAND, OBJ_ID_HAND, OBJ_ID_HAND, 255] # make non-transparent pixels grey 200
    overlay_transparency(mask_hand, original_mask, actual_top_left[0], actual_top_left[1])


def process_ade_input(sample_list_json, hand_dataset):
    maker = AdeHandDatasetMaker(SAVE_METADATA_DIR, SAVE_DATASET_NAME, SAVE_VERSION_NAME)
    for sample_entry in tqdm(sample_list_json):
        img_path = sample_entry['fpath_img']
        seg_path = sample_entry['fpath_segm']
        img = cv2.imread(img_path)
        seg = cv2.imread(seg_path)
        bboxes = get_random_bboxes(seg, display_rgb=img)
        if len(bboxes) == 0:
            continue
        img_hand = img.copy()
        _, hand_patches, patch_ul_corners, fingertip_locations = paste_hand_onto_img(img_hand, seg, bboxes[0], hand_dataset)
        hand_patch = hand_patches[0].copy()
        seg_hand = seg.copy()
        paste_hand_onto_mask(hand_patch, seg_hand, patch_ul_corners[0])
        maker.save_images_and_masks(img, img_hand, seg, seg_hand, {'obj_bbox': bboxes[0][:4], 'obj_class': bboxes[0][4],
                                    'finger_x': fingertip_locations[0][0], 'finger_y': fingertip_locations[0][1]}) # add fingertip locs
        # cv2.imshow('img', img_hand)
        # cv2.waitKey(0)
        # quit()

def process_voc_input(sample_list_json):
    voc_dataset = SBDClassSeg()
    maker = AdeHandDatasetMaker(SAVE_METADATA_DIR, SAVE_DATASET_NAME, SAVE_VERSION_NAME)
    for i in tqdm(range(len(voc_dataset))):
        img, seg = voc_dataset.get_example(i)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        seg = np.uint8(seg)
        bboxes = get_random_bboxes(seg, voc=True)
        seg = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)
        if len(bboxes) == 0:
            continue
        for bbox in bboxes:
            img_with_hand = img.copy()
            _, hand_patches, patch_ul_corners, fingertip_locations = paste_hand_onto_img(img_with_hand, bbox,
                                                                                         hand_dataset)
            hand_patch = hand_patches[0].copy()
            seg_with_hand = seg.copy()
            paste_hand_onto_mask(hand_patch, seg_with_hand, patch_ul_corners[0])
            new_metadata_entry = {'obj_bbox': bbox[:4], 'obj_class': bbox[4],
                                                                       'finger_x': fingertip_locations[0][0],
                                                                       'finger_y': fingertip_locations[0][1]}
            maker.save_images_and_masks(img, img_with_hand, seg, seg_with_hand, new_metadata_entry)  # add fingertip locs
            # print(new_metadata_entry)
            # cv2.imshow('img', img_hand)
            # cv2.imshow('label for ' + str(bbox[4]), seg_hand)
            # cv2.waitKey(0)

if __name__ == "__main__":
    print(getcwd())
    hand_dataset = HandDatasetLoader(HAND_DATASET_ROOT)
    process_voc_input(hand_dataset)
    quit()
    # parse ade image and segmentation image paths and metadata
    gt_metadata_file = path.join(ADE_DATASET_ROOT, "training.odgt")
    input_list = parse_ade_input_list(odgt=gt_metadata_file)  # max_sample=
    # parse hand image paths and metadata

    process_ade_input(input_list, hand_dataset)
