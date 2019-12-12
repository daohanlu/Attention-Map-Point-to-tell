from os import path, mkdir
import json
import signal
import logging
import cv2

from .my_dataset_maker import MyDatasetMaker


class AdeHandDatasetMaker(MyDatasetMaker):
    def __init__(self, save_dir_root, dataset_name, version_name, image_type=".png"):
        super().__init__(save_dir_root, dataset_name, version_name, image_type)
        self.mask_folder_name = self.image_folder_name + "_mask"
        self.mask_filename_prefix = path.join(save_dir_root, self.mask_folder_name, dataset_name + "_")
        self.mask_metadata_path = path.join(self.save_dir_root, self.mask_folder_name + "_metadata.json")
        self.mask_annotations = []
        # make folder that stores masks
        if not path.exists(path.join(save_dir_root, self.mask_folder_name)):
            mkdir(path.join(save_dir_root, self.mask_folder_name))
        if path.exists(self.mask_metadata_path):
            print("Mask metadata file already exists. Type 'a' to append and 'o' to overwrite")
            if input() == 'o':
                self._mask_metadata_writer = open(self.mask_metadata_path, 'w')
            else:
                self._mask_metadata_writer = open(self.mask_metadata_path, 'a')
        else:
            self._mask_metadata_writer = open(self.mask_metadata_path, 'w')

    def _get_new_img_filename(self, is_img_original):
        if is_img_original:
            # do not increment counter when saving the extra copy that is the original img
            return self.image_filename_prefix + str(self._file_counter) + "_o" + ".jpg"
        else:
            return self.image_filename_prefix + str(self._file_counter) + ".jpg"

    def _get_new_mask_filename(self, counter, is_mask_original):
        if is_mask_original:
            return self.mask_filename_prefix + str(counter) + "_o" + ".png"
        else:
            return self.mask_filename_prefix + str(counter) + ".png"

    def save_images_and_masks(self, img_original, img_hand, mask_original, mask_hand, annotation):
        """
        Saves two images and and two masks and add one annotation entry to the metadata file.
        :param img_original: The original image from the ADE20K dataset
        :param img_hand: The image with a hand overlayed on top of the original image
        :param mask_original: The original mask image from the ADE20K dataset
        :param mask_hand: The mask with a mask of the hand overlayed on to of original mask image.
        :param annotation: An annotation entry that is a map with properties including bounding box locations
         and categories for each object appearing in the image.
        """
        self._increment_counter()
        img_save_path = self._get_new_img_filename(is_img_original=False)
        original_img_save_path = self._get_new_img_filename(is_img_original=True)
        mask_save_path = self._get_new_mask_filename(self._file_counter, is_mask_original=False)
        original_mask_save_path = self._get_new_mask_filename(self._file_counter, is_mask_original=True)

        with DelayedKeyboardInterrupt():
            cv2.imwrite(img_save_path, img_hand)
            cv2.imwrite(original_img_save_path, img_original)
            cv2.imwrite(mask_save_path, mask_hand)
            cv2.imwrite(original_mask_save_path, mask_original)
            if annotation is not None and 'fpath_img' in annotation.keys():
                assert annotation['fpath_img'] == "", \
                    "the file path in image annotation should be set by the dataset maker!"
            annotation['fpath_img'] = img_save_path[img_save_path.find(self.save_dir_root) + len(self.save_dir_root) + 1:]
            annotation['has_hand'] = True
            self.annotations.append(annotation)
            self._metadata_writer.write(json.dumps(annotation) + "\n") # write meta for image with hand
            annotation = annotation.copy()
            annotation['fpath_img'] = original_img_save_path[original_img_save_path.find(self.save_dir_root) + len(self.save_dir_root) + 1:]
            annotation['has_hand'] = False
            self.annotations.append(annotation)
            self._metadata_writer.write(json.dumps(annotation) + "\n") # write meta for image without hand

            mask_annotation = annotation.copy()
            mask_annotation['fpath_img'] = mask_save_path[mask_save_path.find(self.save_dir_root) + len(self.save_dir_root) + 1:]
            mask_annotation['has_hand'] = True
            self.mask_annotations.append(mask_annotation)
            self._mask_metadata_writer.write(json.dumps(mask_annotation) + "\n")
            mask_annotation = annotation.copy()
            mask_annotation['fpath_img'] = original_mask_save_path[original_mask_save_path.find(self.save_dir_root) + len(self.save_dir_root) + 1:]
            mask_annotation['has_hand'] = False
            self.mask_annotations.append(mask_annotation)
            self._mask_metadata_writer.write(json.dumps(mask_annotation) + "\n")


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