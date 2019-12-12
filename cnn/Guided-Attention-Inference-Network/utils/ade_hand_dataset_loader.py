import json
from os import path
import random
from glob import glob

import cv2
import random
import numpy as np

TRAIN_PERCENTAGE = 0.85
# VAL_PERCENTAGE = 0.15
SEED = 2019


class AdeHandDatasetLoader():

    def __init__(self, hand_dataset_root, split):
        self.split = split
        self.hand_dataset_root = hand_dataset_root
        # random.seed(SEED)
        self._parse_input_list()
        self._filter_items_by_split()
        self.contained_class = set()

    def _parse_input_list(self, max_sample=-1, start_idx=-1, end_idx=-1):
        """
        Reads a json file to produce a list of hand images samples including a path and metadata
        each entry in self.list_samples will be formatted as the following:
        {"fpath_img": "hands_data/hands_v1/hand_1.png", "width": 1920, "height": 1080, "fingertip_x": 38, "fingertip_y": 0,
            "fingertip_sensor_x": 0.98, "fingertip_sensor_y": 0.7, "focal_length": 4.03, "sensor_width": 4.096, "sensor_height": 2.304}
        """
        metadata_files = glob(path.join(self.hand_dataset_root, "*.json"))
        image_metadata_file = None
        label_metadata_file = None
        for f in metadata_files:
            if f.find('mask') < 0:
                image_metadata_file = f
            else:
                label_metadata_file = f
        list_image = [json.loads(x.rstrip()) for x in open(image_metadata_file, 'r')]
        list_labels = [json.loads(x.rstrip()) for x in open(label_metadata_file, 'r')]
        assert len(list_image) == len(list_labels) and len(list_image) > 0

        if max_sample > 0:
            list_image = list_image[0:max_sample]
            list_labels = list_labels[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:  # divide file list
            list_image = list_image[start_idx:end_idx]
            list_labels = list_labels[start_idx:end_idx]
        self.list_image = list_image
        self.list_labels = list_labels
        assert len(list_image) == len(list_labels)

    def _filter_items_by_split(self):
        # self.list_image = random.shuffle(self.list_image)
        # self.list_labels = random.shuffle(self.list_labels)
        size = self.__len__()
        if self.split == 'train':
            self.list_image = self.list_image[:int(TRAIN_PERCENTAGE * size)]
            self.list_labels = self.list_labels[:int(TRAIN_PERCENTAGE * size)]
        elif self.split == 'val':
            self.list_image = self.list_image[int(TRAIN_PERCENTAGE * size):]
            self.list_labels = self.list_labels[int(TRAIN_PERCENTAGE * size):]
        else:
            raise ValueError("'split' argument to __init__ must be either 'train' or 'val'!")
        print('#{} ade-hand samples: {}'.format(self.split, self.__len__()))

    def filter_by_obj_class(self, classes):
        classes = set(classes)
        self.contained_class = classes

        newlist = []
        for item in self.list_image:
            if item['obj_class'] in classes:
                newlist.append(item)
        self.list_image = newlist
        del newlist

        newlist = []
        for item in self.list_labels:
            if item['obj_class'] in classes:
                newlist.append(item)
        self.list_labels = newlist
        print('#{} ade-hand samples after filter: {}'.format(self.split, self.__len__()))
        assert len(self.list_image) == len(self.list_labels)

    def get_image_path(self, i):
        # list_samples[0] contains the image paths
        # p = self.list_image[i]['fpath_img']
        # p = path.dirname(p)
        # return p
        return self.list_image[i]['fpath_img']

    def get_label_path(self, i):
        # list_samples[0] contains the segmentation image / label paths
        return self.list_labels[i]['fpath_img']

    def get_image_metadata(self, i):
        return self.list_image[i]

    def __len__(self):
        return len(self.list_image)