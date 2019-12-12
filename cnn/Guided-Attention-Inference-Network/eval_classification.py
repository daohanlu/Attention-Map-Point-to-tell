import os
import argparse
import chainer
import cv2
from chainer import Variable
import chainer.functions as F
from models.fcn8_hand_v2 import FCN8s_hand
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainer.iterators import SerialIterator
from chainer.serializers import load_npz
from chainer.backends.cuda import get_array_module
from chainercv.datasets.voc.voc_utils import voc_semantic_segmentation_label_names

from matplotlib import pyplot as plt
import numpy as np
import cupy as cp

from my_training_dataset import MyTrainingDataset
from postprocessing import gcams_to_bboxes, gcams_to_mask

def printSummary(tp, tn, fp, fn):
    pass

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pretrained', type=str,
                        help='path to model that has trained classifier but has not been trained through GAIN routine',
                        default='classifier_padding_1_model_594832')
    parser.add_argument('--trained', type=str, help='path to model trained through GAIN',
                        default='result/MYGAIN_5_to_1_padding_1_all_update_model_20000')
    parser.add_argument('--device', type=int, default=0, help='gpu id')
    parser.add_argument('--shuffle', type=bool, default=False, help='whether to shuffle dataset')
    parser.add_argument('--whole', type=bool, default=False, help='whether to test for the whole validation dataset')
    parser.add_argument('--no', type=int, default=5, help='if not whole, then no of images to visualize')
    parser.add_argument('--name', type=str, default='viz1', help='name of the subfolder or experiment under which to save')

    args = parser.parse_args()

    pretrained_file = args.pretrained
    trained_file = args.trained
    device = args.device
    shuffle = args.shuffle
    whole = args.whole
    name = args.name
    N = args.no

    dataset = MyTrainingDataset(split='val')
    iterator = SerialIterator(dataset, 1, shuffle=shuffle, repeat=False)
    converter = chainer.dataset.concat_examples
    os.makedirs('viz/' + name, exist_ok=True)
    no_of_classes = 21
    device = 0
    pretrained = FCN8s_hand()
    trained = FCN8s_hand()
    load_npz(pretrained_file, pretrained)
    load_npz(trained_file, trained)

    if device >= 0:
        pretrained.to_gpu()
        trained.to_gpu()
    i = 0

    true_positive = [0 for j in range(21)]
    true_negative = [0 for j in range(21)]
    false_positive = [0 for j in range(21)]
    false_negative = [0 for j in range(21)]

    while not iterator.is_new_epoch:

        if not whole and i >= N:
            break

        image, labels, metadata = converter(iterator.next())
        np_input_img = image
        np_input_img = np.uint8(np_input_img[0])
        np_input_img = np.transpose(np_input_img, (1,2,0))
        image = Variable(image)
        if device >= 0:
            image.to_gpu()

        xp = get_array_module(image.data)
        to_substract = np.array((-1, 0))
        noise_classes = np.unique(labels[0]).astype(np.int32)
        target = xp.asarray([[0] * (no_of_classes)])
        gt_labels = np.setdiff1d(noise_classes, to_substract) - 1
        target[0][gt_labels] = 1


        gcam1, cl_scores1, class_id1 = pretrained.stream_cl(image)
        gcam2, cl_scores2, class_id2 = trained.stream_cl(image)
        # gcams1, cl_scores1, class_ids1 = pretrained.stream_cl_multi(image)
        # gcams2, cl_scores2, class_ids2 = trained.stream_cl_multi(image)

        target = cp.asnumpy(target)
        cl_scores2 = cp.asnumpy(cl_scores2.data)
        # print(target)
        # print(cl_scores2)
        # print()
        # score_sigmoid = F.sigmoid(cl_scores2)
        for j in range(0, len(target[0])):
            # print(target[0][j] == 1)
            if target[0][j] == 1:
                if cl_scores2[0][j] >= 0:
                    true_positive[j] += 1
                else:
                    false_negative[j] += 1
            else:
                if cl_scores2[0][j] <= 0:
                    true_negative[j] += 1
                else:
                    false_positive[j] += 1
        # bboxes = gcams_to_bboxes(gcams2, class_ids2, input_image=np_input_img)
        # cv2.imshow('input', np_input_img)
        # cv2.waitKey(0)
        if device > -0:
            class_id = cp.asnumpy(class_id)
        # fig1 = plt.figure(figsize=(20, 10))
        # ax1 = plt.subplot2grid((3, 9), (0, 0), colspan=3, rowspan=3)
        # ax1.axis('off')
        # ax1.imshow(cp.asnumpy(F.transpose(F.squeeze(image, 0), (1, 2, 0)).data) / 255.)
        #
        # ax2 = plt.subplot2grid((3, 9), (0, 3), colspan=3, rowspan=3)
        # ax2.axis('off')
        # ax2.imshow(cp.asnumpy(F.transpose(F.squeeze(image, 0), (1, 2, 0)).data) / 255.)
        # ax2.imshow(cp.asnumpy(F.squeeze(gcam1[0], 0).data), cmap='jet', alpha=.5)
        # ax2.set_title("Before GAIN for class - " + str(dataset.class_names[cp.asnumpy(class_id1)+1]),
        #               color='teal')
        #
        # ax3 = plt.subplot2grid((3, 9), (0, 6), colspan=3, rowspan=3)
        # ax3.axis('off')
        # ax3.imshow(cp.asnumpy(F.transpose(F.squeeze(image, 0), (1, 2, 0)).data) / 255.)
        # ax3.imshow(cp.asnumpy(F.squeeze(gcam2[0], 0).data), cmap='jet', alpha=.5)
        # ax3.set_title("After GAIN for class - " + str(dataset.class_names[cp.asnumpy(class_id2)+1]),
        #               color='teal')
        # fig1.savefig('viz/' + name + '/' + str(i) + '.png')
        # plt.close()
        print(i)
        i += 1
    print("true postive {}".format(true_positive))
    print("true negative {}".format(true_negative))
    print("false positive {}".format(false_positive))
    print("false negative {}".format(false_negative))

if __name__ == "__main__":
    main()