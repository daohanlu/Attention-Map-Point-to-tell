import json
from os import path
import random
from glob import glob

import cv2

BINS_PER_AXIS = 10


def _location_to_bin_index(rel_x, rel_y):
    x_bin_index = int((rel_x - 0.00001) * BINS_PER_AXIS)  # subtract to avoid index out of bounds
    y_bin_index = int((rel_y - 0.00001) * BINS_PER_AXIS)
    return x_bin_index, y_bin_index


class HandDatasetLoader:

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

    def bin_samples_by_location(self):
        """
        Put each sample in to one of numbins*numbins bins by its location so we can get a random hand sample by
        its relative location in the future
        :param numbins: number of bins for both the x and y axis
        """
        print("hand_dataset_loader.py: using {}*{} bins for hand locations".format(len(self.bins_of_samples),
              len(self.bins_of_samples[0])))
        for entry in self.list_samples:
            try:
                x_bin_index, y_bin_index = _location_to_bin_index(entry['rel_loc_x'], entry['rel_loc_y'])
            except KeyError:
                x_bin_index, y_bin_index = _location_to_bin_index(entry['fingertip_sensor_x'], entry['fingertip_sensor_y'])

            # print(str(entry['fingertip_sensor_x']) + " -> " + str(x_bin_index))
            # print(str(entry['fingertip_sensor_y']) + " -> " + str(y_bin_index))

            self.bins_of_samples[x_bin_index][y_bin_index].append(entry)


    def get_sample_list(self):
        return self.list_samples

    def get_random_hand_by_location(self, rel_x, rel_y):
        """
        Returns a metadata entry of the hand in the bin that covers rel_x, rel_y
        :param rel_x: relative x position on screen, expressed as a ratio: x / frame-width
        :param rel_y: relative x position on screen, expressed as a ratio: y / frame-height
        :return: metadata entry of the hand in the bin that covers rel_x, rel_y
        """
        x_bin_index, y_bin_index = _location_to_bin_index(rel_x, rel_y)
        x_bin_index = min(x_bin_index, BINS_PER_AXIS - 1)
        y_bin_index = min(y_bin_index, BINS_PER_AXIS - 1)
        if len(self.bins_of_samples[x_bin_index][y_bin_index]) > 0:
            return random.sample(self.bins_of_samples[x_bin_index][y_bin_index], 1)[0]
        else:
            non_empty_bin = []
            while len(non_empty_bin) == 0:
                bin_x = random.randint(0, BINS_PER_AXIS - 1)
                bin_y = random.randint(0, BINS_PER_AXIS - 1)
                non_empty_bin = self.bins_of_samples[bin_x][bin_y]
            return random.sample(non_empty_bin, 1)[0]
            direction_x = random.choice([-1, 1])
            direction_y = random.choice([-1, 1])
            sample = None
            for i in range(BINS_PER_AXIS):
                try:
                    sample = random.sample(self.bins_of_samples[x_bin_index + direction_x * i][y_bin_index], 1)[0]
                except ValueError:
                    continue
                except IndexError:
                    break
                if sample is not None:
                    return sample
            return None

    def get_fingertip_loc_by_metadata(self, metadata_entry):
        return metadata_entry['fingertip_x'], metadata_entry['fingertip_y']

    def get_image_by_metadata(self, metadata_entry):
        img = cv2.imread(metadata_entry['fpath_img'], cv2.IMREAD_UNCHANGED)
        return img
