import chainercv
import cv2
import numpy as np
import os
import chainer
from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.datasets.voc import voc_utils
from chainercv.utils import read_image
from chainercv.utils import read_label
import PIL

from utils.ade_hand_dataset_loader import AdeHandDatasetLoader

DATA_DIR = '/home/mmvc/Git/point-to-tell/VocHand_3'
INFER_DATA_DIR = '/home/mmvc/Git/point-to-tell/VocHand_infer'
OBJ_IDS_OF_INTEREST = {5, 9, 11, 15, 18, 20}  # bottle chair table person sofa tv/monitor

class MyTrainingDataset(chainer.dataset.DatasetMixin):

    """Semantic segmentation dataset for PASCAL `VOC2012`_.

    .. _`VOC2012`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/voc`.
        split ({'train', 'val', 'trainval'}): Select a split of the dataset.

    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"
        :obj:`label`, ":math:`(H, W)`", :obj:`int32`, \
        ":math:`[-1, \#class - 1]`"
    """
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

    def __init__(self, split='train'):
        super(MyTrainingDataset, self).__init__()
        self.split = split
        if split not in ['train', 'trainval', 'val']:
            raise ValueError(
                'please pick split from \'train\', \'trainval\', \'val\'')
        if split == 'val':
            self.data_dir = INFER_DATA_DIR
        else:
            self.data_dir = DATA_DIR
        self.loader = AdeHandDatasetLoader(self.data_dir, split)
        self.loader.filter_by_obj_class(OBJ_IDS_OF_INTEREST) # keep only: person, table, chair
        assert self.loader.__len__() > 0

        # self.add_getter('img', self._get_image)
        # self.add_getter('label', self._get_label)

    def __len__(self):
        return self.loader.__len__()

    def get_example(self, index):
        # data_file = self.files[self.split][index]
        # # load image
        # img_file = data_file['img']
        # img = PIL.Image.open(img_file)
        # img = np.array(img, dtype=np.uint8)
        # # load label
        # lbl_file = data_file['lbl']
        # lbl = PIL.Image.open(lbl_file)
        # lbl = np.array(lbl, dtype=np.int32)
        # lbl[lbl == 255] = -1
        # print(self.loader.get_image_metadata(index))
        return self._get_image(index), self._get_label(index), np.array(self.loader.get_image_metadata(index))

    def _get_image(self, i):
        img_path = os.path.join(
            self.data_dir, self.loader.get_image_path(i))
        # img = cv2.imread(img_path).astype(np.float32)
        img = read_image(img_path, color=True)
        if self.split != 'val':
            img = chainercv.transforms.pca_lighting(img, 25.5)
            img = chainercv.transforms.random_flip(img, x_random=True, y_random=False)  # random horizontal flip
        # visualization
        # img = np.array(img, dtype=np.uint8)
        # img = np.transpose(img, (2, 1, 0))
        # print(img)
        # print(img.shape)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        return img

    def _get_label(self, i):
        label_path = os.path.join(
            self.data_dir, self.loader.get_label_path(i))
        label = PIL.Image.open(label_path)
        # print(np.array(label))
        # input()
        label = np.array(label, dtype=np.int32)
        # print(np.count_nonzero(label > 0))
        # label[label == 255] = -1
        label = label[..., 0]
        # label = np.expand_dims(label, 2)
        # label = label[..., 0] # to gray
        # print(np.count_nonzero(label > 0))
        # cv2.imshow('lbl', np.uint8(label * 10))
        # cv2.waitKey(0)
        # (1, H, W) -> (H, W)
        return label

# self-test
if __name__ == '__main__':
    train_set = MyTrainingDataset(split='train')
    val_set = MyTrainingDataset(split='val')
    print('first image from training split {}'.format(train_set.get_example(0)[2]))
    print('first image from val split {}'.format(val_set.get_example(0)[2]))