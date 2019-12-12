import glob
from os import path, mkdir
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from utils.my_dataset_maker import MyDatasetMaker
from utils.color_temp import convert_temperature

METADATA_DIR = "hands_data"
VERSION_NAME = "v4"
DATASET_NAME = "hands_right"
SAVE_DIR = path.join(METADATA_DIR, DATASET_NAME + "_" + VERSION_NAME)

save_file_counter = 0


def save_foreground_img(foreground_img, filename):
    """
    Saves the foreground image to disk with the background being transparent
    :param foreground_img: the image of the hand in that has been cropped, with background pixels strictly black
    """
    background_mask = np.max(foreground_img[..., :], 2) == 0
    foreground_img = cv2.cvtColor(foreground_img, cv2.COLOR_RGB2BGRA)
    foreground_img[background_mask, 3] = 0
    cv2.imwrite(filename, foreground_img)


def autocrop(image, vertical_threshold=60, horizontal_threshold=30):
    """Crops any edges below or equal to threshold
    Returns cropped image.
    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    crop_upper_left = (0,0)
    nonblack_row_indices = np.where(np.max(flatImage, 1) > vertical_threshold)[0]
    if nonblack_row_indices.size:
        # eliminate useless rows first
        image = image[nonblack_row_indices[0]: nonblack_row_indices[-1] + 1, :]
        # then crop columns
        nonblack_col_indices = np.where(np.max(image, 0) > horizontal_threshold)[0]
        image = image[:, nonblack_col_indices[0]: nonblack_col_indices[-1] + 1]
        crop_upper_left = (nonblack_col_indices[0], nonblack_row_indices[0])
    else:
        image = image[:1, :1]
    return image, crop_upper_left


def remove_background(image, lower_clr_bound, upper_clr_bound):
    image = convert_temperature(image, temp=7000)
    image_copy = np.copy(image)
    # convert original to RGB
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    # if(save_file_counter > 100):
    #     plt.imshow(image_copy)
    #     plt.show()
    # reserve true blacks as for mask
    # create mask
    # mask_background = cv2.inRange(image_copy, lower_clr_bound, upper_clr_bound)
    r_channel = image_copy[..., 0].astype(int)
    g_channel = image_copy[..., 1].astype(int)
    b_channel = image_copy[..., 2].astype(int)
    rg_minus_2b = (r_channel + g_channel - b_channel * 2)
    # plt.imshow(r_channel)
    # plt.show()
    # plt.imshow(b_channel)
    # plt.show()
    # plt.imshow(rb_difference, cmap='gray')
    # plt.show()
    # use mask
    masked_image = np.copy(image_copy)
    # background: r+g - 2b is very negative and b channel is bright
    masked_image[(rg_minus_2b < -19*2) | (b_channel > 190)] = [0, 0, 0]
    mask_edges = masked_image[..., 1] > masked_image[..., 0]
    masked_image[mask_edges] = [0, 0, 0]
    cropped_image, crop_upper_left = autocrop(masked_image)
    # if (save_file_counter > 100):
    #     plt.imshow(cropped_image)
    #     plt.show()
    return cropped_image, crop_upper_left


def calculate_fingertip_loc(foreground_image):
    """we know the foreground's first row is non-black. We can assume the finger's y index is 0.
    So calculate the center of brightness of the first row to get the x index"""
    flatImage = np.max(foreground_image, 2)
    first_row = flatImage[0][:]
    brightness_center = np.dot(np.arange(0, len(first_row)), first_row) / np.sum(first_row)
    return int(brightness_center), 0


def get_filename(save_dir, file_counter):
    return path.join(save_dir, DATASET_NAME + "_" + str(file_counter) + ".png")


def update_metadata(metadata, filename, width, height, fingertip_loc_in_crop, fingertip_loc_in_frame):
    rel_loc_x = round(fingertip_loc_in_frame[0] / width, 3)
    rel_loc_y = round(fingertip_loc_in_frame[1] / height, 3)
    new_entry = {"fpath_img": "", "width": width, "height": height, "fingertip_x": fingertip_loc_in_crop[0],
                 "fingertip_y": fingertip_loc_in_crop[1], "rel_loc_x": rel_loc_x, "rel_loc_y": rel_loc_y,
                 "focal_length": 4.03, "sensor_width": 4.096, "sensor_height": 2.304}
    metadata.append(new_entry)


def get_last_annotation(metadata_list):
    return metadata_list[-1]


def main():
    dataset_maker = MyDatasetMaker(METADATA_DIR, DATASET_NAME, VERSION_NAME)
    video_files = glob.glob('data/*.mp4')
    cap = cv2.VideoCapture(video_files[0])
    lower_green = np.array([0, 120, 0])     ##[R value, G value, B value]
    upper_green = np.array([100, 255, 100])
    _, original_img = cap.read()

    image_metadata_list = []
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Found " + str(length) + " frames in input video")
    pbar = tqdm.tqdm(total=length)
    # quit()
    global save_file_counter
    for i in range(40):
        _, __ = cap.read()
    try:
        while cap.isOpened():
            _, original_img = cap.read()
            if len(original_img) < 48 or len(original_img[0]) < 48:
                break
            foreground, crop_ul = remove_background(original_img, lower_green, upper_green)
            # ignore potential hands with size smaller than 48*48
            if len(foreground) < 48 or len(foreground[0]) < 48:
                continue
            save_file_counter += 1
            finger_x, finger_y = calculate_fingertip_loc(foreground)
            fingertip_loc_in_frame = (crop_ul[0] + finger_x, crop_ul[1] + finger_y)
            # cv2.rectangle(foreground, (finger_x, finger_y), (finger_x + 12, finger_y + 12), (255, 0, 0), 4)
            # cv2.rectangle(original_img, (crop_ul[0] + finger_x, crop_ul[1] + finger_y), (crop_ul[0] + finger_x + 12, crop_ul[1] + finger_y + 12), (255, 0, 0), 4)
            filename = get_filename(SAVE_DIR, save_file_counter)
            # save_foreground_img(foreground, filename)
            update_metadata(image_metadata_list, filename, original_img.shape[1], original_img.shape[0], (finger_x, finger_y), fingertip_loc_in_frame)
            dataset_maker.save_foreground_image(foreground, get_last_annotation(image_metadata_list))
            pbar.update(1)
                #print(json.dumps(entry))
            # plt.imshow(original_img)
            # plt.show()

    except (KeyboardInterrupt, SystemExit):
        print("Aborting ground truth generation. Saving partial metadata...")
    except TypeError:
        print("Video frames ended prematurely. Saving partial metadata...")
    #process unkown err

    # with open(SAVE_DIR + '_metadata.json', 'w') as outfile:
    #     for entry in image_metadata_list:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')


if __name__ == "__main__":
    main()