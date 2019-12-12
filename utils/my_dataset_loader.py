import json
from os import path
import random
from glob import glob

import cv2



class MyDataSetLoader:

    def __init__(self, hand_dataset_root):
        self.hand_metadata_filenames = glob(path.join(hand_dataset_root, "*.json"))
        self.num_sample = 0
        self.list_samples = []
        # a 2-D array that groups hand samples into n*n bins by their relative locations
        self.bins_of_samples = [[[] for i in range(BINS_PER_AXIS)] for j in range(BINS_PER_AXIS)]
        self.parse_input_list(self.hand_metadata_filenames)

    def parse_input_list(self, json_files, max_sample=-1, start_idx=-1, end_idx=-1):
        """
        Reads a json file to produce a list of hand images samples including a path and metadata
        each entry in self.list_samples will be formatted as the following:
        {"fpath_img": "hands_data/hands_v1/hand_1.png", "width": 1920, "height": 1080, "fingertip_x": 38, "fingertip_y": 0,
            "fingertip_sensor_x": 0.98, "fingertip_sensor_y": 0.7, "focal_length": 4.03, "sensor_width": 4.096, "sensor_height": 2.304}
        """
        list_sample = []
        for json_file in json_files:
            list_sample += [json.loads(x.rstrip()) for x in open(json_file, 'r')]
        if max_sample > 0:
            list_sample = list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:  # divide file list
            list_sample = list_sample[start_idx:end_idx]
        self.num_sample = len(list_sample)
        assert self.num_sample > 0
        self.list_samples = list_sample
        self.bin_samples_by_location()
        print('# hand samples: {}'.format(self.num_sample))

    def get_sample_list(self):
        return self.list_samples

    def get_image_by_metadata(self, metadata_entry):
        img = cv2.imread(metadata_entry['fpath_img'], cv2.IMREAD_UNCHANGED)
        #print(img.shape)
        return img
