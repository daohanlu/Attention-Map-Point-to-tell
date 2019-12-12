import os
import argparse
import chainer
import cv2
from chainer import Variable
import chainer.functions as F

from chainercv.datasets import VOCSemanticSegmentationDataset
from chainer.iterators import SerialIterator
from chainer.serializers import load_npz
from chainer.backends.cuda import get_array_module

from matplotlib import pyplot as plt
import numpy as np
import cupy as cp
from PIL import Image

from models.fcn8 import FCN8s
from models.fcn8_hand_v2 import FCN8s_hand
from my_training_dataset import MyTrainingDataset
from postprocessing import gcams_to_mask
from chainercv.utils import read_image

# MODEL_PATH = "/home/mmvc/Git/point-to-tell/cnn/Guided-Attention-Inference-Network/result/MYGAIN_dropout_5_to_1_model_10000"
# MODEL_PATH = "/home/mmvc/Git/point-to-tell/cnn/Guided-Attention-Inference-Network/result/classifier_gain_dropout_model_88592"
# MODEL_PATH = "/home/mmvc/Git/point-to-tell/cnn/Guided-Attention-Inference-Network/classifier_gain_dropout_model_303744"
# MODEL_PATH = "/home/mmvc/Git/point-to-tell/cnn/Guided-Attention-Inference-Network/classifier_padding_1_model_594832"
MODEL_PATH = "/home/mmvc/Git/point-to-tell/cnn/Guided-Attention-Inference-Network/result/MYGAIN_5_to_1_padding_1_all_update_model_20000"

# VIDEO_PATH = "/home/mmvc/Downloads/Telegram Desktop/20191113_232549.mp4"
VIDEO_PATH = "/home/mmvc/Videos/C0009.MP4"
GPU_DEVICE = 0
FRAMES_PER_INFERENCE = 10

converter = chainer.dataset.concat_examples

def chainer_img_to_np_img(chainer_img):
    image_np = chainer_img[:, :, ::-1]  # BGR -> RGB
    image_np = image_np.transpose((1, 2, 0))  # HWC -> CHW
    return image_np

def show_multi_gcam(model, image_np):
    print("new img!")
    # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_np = cv2.resize(image_np, (500,261), cv2.INTER_CUBIC)
    original_image = image_np.copy()
    if image_np.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        image_np = image_np[np.newaxis].astype(dtype)
    else:
        # alpha channel is included
        image_np = image_np[:, :, ::-1]  # BGR -> RGB
        image_np = image_np.transpose((2, 0, 1))  # HWC -> CHW

    """
    # path = "/home/mmvc/Git/point-to-tell/VocHand_3/VOCHand_v3/VOCHand_7377_o.jpg"
    path = '/home/mmvc/Documents/cars.jpg'
    path = "/home/mmvc/Git/point-to-tell/hands_data/hands_right_v3/hands_right_16.png"
    path = "/home/mmvc/Git/point-to-tell/hands_data/hands_right_v1/hand_371.png"
    path = "/home/mmvc/Git/point-to-tell/hands_data/hands_right_v2/hands_right_1.png"
    path = "/home/mmvc/Git/point-to-tell/hands_data/hands_right_v1/hand_936.png"
    path = '/home/mmvc/Pictures/vlcsnap-2019-11-14-01h21m27s136.png'
    path = '/home/mmvc/Documents/test_photos_1080/_DSC3133.JPG' # no hand
    path = '/home/mmvc/Documents/test_photos_1080/_DSC3148.JPG' # only hand
    path = '/home/mmvc/Documents/test_photos_1080/_DSC3160.JPG' # hand in focus
    path = '/home/mmvc/Documents/test_photos_1080/_DSC3147.JPG' # hand out of focus
    image_np = read_image(path, color=True)
    # image_np = read_image("/home/mmvc/Git/point-to-tell/hands_data/hands_right_v1/hand_371.png", color=True)
    # image_np = read_image("/home/mmvc/Git/point-to-tell/VocHand_3/VOCHand_v3/VOCHand_1.jpg", color=True)
    # image_np = read_image("/home/mmvc/Documents/hand.jpg", color=True)
    original_image = cv2.imread(path)
    original_image = np.uint32(original_image)
    """

    image_np = np.array([image_np])
    image_np = image_np.astype(np.float32)
    image_chainer = Variable(image_np)
    if GPU_DEVICE >= 0:
        image_chainer.to_gpu()

    gcams, cl_scores, class_ids = model.stream_cl_multi(image_chainer, use_max_pooling=False)
    print(cl_scores)

    cv2.imshow('original', np.uint8(original_image))
    gcam, cl_score, class_id = model.stream_cl(image_chainer)

    # masking based on gcam
    masked_image = model.mask_image(image_chainer, model.get_mask(gcam))
    # img = chainer_img_to_np_img(cp.asnumpy(masked_image[0].data))
    img = np.uint8(chainer_img_to_np_img(cp.asnumpy(masked_image[0].data)))
    masked_output = model.stream_am(masked_image)
    am_loss = F.sigmoid(masked_output[0][class_id])
    print("masked conf stream am: " + str(masked_output))
    print("new conf {} am loss {}".format(masked_output[0][class_id], am_loss))
    cv2.imshow('masked img', img)

    mask = gcams_to_mask(gcams, class_ids, img=original_image)
    if mask is not None and len(mask) > 0:
        final_masked_image = np.uint32(np.floor_divide(original_image, 2))
        mask = np.uint32(np.floor_divide(mask, 1.6))
        final_masked_image += mask
        final_masked_image = np.uint8(final_masked_image)
    else:
        print("No mask!")
        final_masked_image = np.uint8(np.floor_divide(original_image, 2))
    cv2.imshow("mask", final_masked_image)
    cv2.waitKey(0)

def skip_frames(cap, n):
    i = 0
    while i < n and cap.isOpened():
        cap.read()
        i += 1

def main():
    # no_of_classes = 20
    no_of_classes = 21
    # FCN8s()
    # pretrained = FCN8s_hand()
    trained = FCN8s_hand()
    # load_npz(pretrained_file, pretrained)
    # trained = pretrained
    load_npz(MODEL_PATH, trained)

    if GPU_DEVICE >= 0:
        # pretrained.to_gpu()
        trained.to_gpu()
    i = 0

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not os.path.exists(VIDEO_PATH):
        print("video file doesn't exist!")
        quit()
    skip_frames(cap, 3)
    while cap.isOpened():
        _, img_np = cap.read()
        show_multi_gcam(trained, img_np)
        # cv2.imshow("mask", img_np)
        # cv2.waitKey(0)
        skip_frames(cap, FRAMES_PER_INFERENCE)


if __name__ == "__main__":
    main()