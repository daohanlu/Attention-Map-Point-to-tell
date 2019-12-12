from os import path, mkdir
import json
import signal
import logging

import cv2
import numpy as np


class MyDatasetMaker:

    def __init__(self, save_dir_root, dataset_name, version_name, image_type=".png"):
        """
        Creates a new MyDatasetMaker object.
        :param save_dir_root: (String) path under which to save metadata/annotation file and the folder of actual
            images. The images will be named "[dataset_name]v_[version_name]_#.png"
        :param dataset_name: (String) The name for the dataset. For example: "hands_left"
        :param version_name: (String) The version name for the dataset. For example: "v2.1"
        :param image_type: file format in which the images are saved. Default ".png"
        """
        self.save_dir_root = save_dir_root
        self.dataset_name = dataset_name
        self.version_name = version_name
        self.image_type = image_type
        self.image_folder_name = path.join(dataset_name + "_" + version_name)
        self.image_filename_prefix = path.join(save_dir_root, self.image_folder_name, dataset_name + "_")
        self.metadata_path = path.join(self.save_dir_root, self.image_folder_name + "_metadata.json")
        self._file_counter = 0
        self.annotations = []

        self._metadata_writer = None
        if not path.exists(self.save_dir_root):
            mkdir(self.save_dir_root)
        # make folder that stores images
        if not path.exists(path.join(save_dir_root, self.image_folder_name)):
            mkdir(path.join(save_dir_root, self.image_folder_name))
        if path.exists(self.metadata_path):
            print("Metadata file already exists. Type 'a' to append and 'o' to overwrite")
            if input() == 'o':
                self._metadata_writer = open(self.metadata_path, 'w')
            else:
                self._file_counter = self._get_file_counter(open(self.metadata_path, 'r'))
                self._metadata_writer = open(self.metadata_path, 'a')
        else:
            self._metadata_writer = open(self.metadata_path, 'w')

    def _get_file_counter(self, metadata_reader):
        # raise NotImplemented()
        annotations = [json.loads(x.rstrip()) for x in metadata_reader]
        last_filename = annotations[-1]['fpath_img']
        last_file_number = int( last_filename[last_filename.rfind('_') + 1: last_filename.rfind('.')] )
        return last_file_number

    def _increment_counter(self):
        self._file_counter += 1

    def _get_new_img_filename(self):
        return self.image_filename_prefix + str(self._file_counter) + ".png"

    def save_foreground_image(self, foreground_img, annotation):
        """
        Saves the foreground image to disk with the background (any true black pixels) being transparent
        :param foreground_img: the image of the hand in that has been cropped, with background pixels strictly black
        """
        assert self.image_type == ".png", "self.image_type must be '.png' to save images with transparency"
        background_mask = np.max(foreground_img[..., :], 2) == 0
        foreground_img = cv2.cvtColor(foreground_img, cv2.COLOR_RGB2BGRA)
        foreground_img[background_mask, 3] = 0
        img_save_path = self._get_new_img_filename()

        with DelayedKeyboardInterrupt():
            cv2.imwrite(img_save_path, foreground_img)
            assert annotation is None or annotation['fpath_img'] == "",\
                "the file path in image annotation should be set by the dataset maker!"
            annotation['fpath_img'] = img_save_path
            self.annotations.append(annotation)
            self._metadata_writer.write(json.dumps(annotation) + "\n")


class DelayedKeyboardInterrupt(object):
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)