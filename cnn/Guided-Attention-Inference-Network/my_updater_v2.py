import cv2
from chainer.training import StandardUpdater
from chainer import Variable
from chainer import report
from chainer import functions as F
from chainer.backends.cuda import get_array_module
import numpy as np
import cupy as cp

from chainercv.utils import read_image
import random

def chainer_img_to_np_img(chainer_img):
    image_np = chainer_img[:, :, ::-1]  # BGR -> RGB
    image_np = image_np.transpose((1, 2, 0))  # HWC -> CHW
    return image_np


class VOC_ClassificationUpdater_v2(StandardUpdater):
	def __init__(self, iterator, optimizer, no_of_classes=21, device=-1, dropout=0.5):
		super(VOC_ClassificationUpdater_v2, self).__init__(iterator, optimizer)
		self.device = device
		self.no_of_classes=no_of_classes
		self.dropout = dropout
		self._optimizers['main'].target.freeze_conv_layers()

	def evaluate_single(self, model):
		path = '/home/mmvc/Documents/test_photos_1080/_DSC3158.JPG'
		image_np = read_image(path, color=True)
		# image_np = read_image("/home/mmvc/Git/point-to-tell/hands_data/hands_right_v1/hand_371.png", color=True)
		# image_np = read_image("/home/mmvc/Git/point-to-tell/VocHand_3/VOCHand_v3/VOCHand_1.jpg", color=True)
		# image_np = read_image("/home/mmvc/Documents/hand.jpg", color=True)
		original_image = cv2.imread(path)
		original_image = np.uint32(original_image)
		image_np = np.array([image_np])
		image_chainer = Variable(image_np)
		image_chainer.to_gpu()
		cl_scores = model.classify(image_chainer, dropout_ratio=None)
		print("hand: " + str(cl_scores))

		path = '/home/mmvc/Documents/test_photos_1080/_DSC3133.JPG'
		image_np = read_image(path, color=True)
		# image_np = read_image("/home/mmvc/Git/point-to-tell/hands_data/hands_right_v1/hand_371.png", color=True)
		# image_np = read_image("/home/mmvc/Git/point-to-tell/VocHand_3/VOCHand_v3/VOCHand_1.jpg", color=True)
		# image_np = read_image("/home/mmvc/Documents/hand.jpg", color=True)
		original_image = cv2.imread(path)
		original_image = np.uint32(original_image)
		image_np = np.array([image_np])
		image_chainer = Variable(image_np)
		image_chainer.to_gpu()
		cl_scores = model.classify(image_chainer, dropout_ratio=None)
		print("no hand: " + str(cl_scores))

	def update_core(self):
		example = self.get_iterator('main').next()
		assert len(example) == 1  # batchsize 1
		image, labels, metadata = self.converter(example)
		# cv2.imshow('img', np.uint8(labels[0].data))
		image = Variable(image)

		if self.device >= 0:
			image.to_gpu(self.device)
		cp.cuda.Device(self.device).use()
		cl_output = self._optimizers['main'].target.classify(image, dropout_ratio=self.dropout)
		xp = get_array_module(cl_output.data)
		interest_mask = xp.asarray([[0] * (self.no_of_classes)] * cl_output.shape[0])
		# ignore objects we don't care about
		for obj_id in [5, 9, 11, 15, 18, 20, 21]:
			interest_mask[..., obj_id - 1] = 1

		target = xp.asarray([[0]*(self.no_of_classes)]*cl_output.shape[0])
		for i in range(labels.shape[0]):
			gt_labels = np.unique(np.uint8(labels[i].data))[1:] - 1  # Not considering 0
			# print(gt_labels )
			target[i][gt_labels] = 1

			# gt_labels = np.unique(labels[i]).astype(np.int32)[2:] - 1 # Not considering -1 & 0
			# gt_labels = xp.asarray([])
			# if not metadata[0]['has_hand']:
			# 	gt_labels = xp.asarray([0])
			# else:
			# 	gt_labels = xp.asarray([metadata[0]['obj_class'], 21])
			# target[i][gt_labels] = 1
		# print(target)
		# print(cl_output)
		# print(metadata)
		# cv2.waitKey(0)
		# loss = F.sigmoid_cross_entropy(cl_output[interest_mask], target[interest_mask], normalize=True)
		loss = F.sigmoid_cross_entropy(cl_output, target, normalize=True)
		report({'Loss':loss}, self.get_optimizer('main').target)
		self._optimizers['main'].target.cleargrads()
		loss.backward()
		self._optimizers['main'].update()
		# if random.random() < 0.02:
		# 	self.evaluate_single(self._optimizers['main'].target)



class VOC_GAIN_Updater_v2(StandardUpdater):

	def __init__(self, iterator, optimizer, no_of_classes=21, device=-1, lambd1=1.5, lambd2=1, lambd3=1.5, dropout=None):
		super(VOC_GAIN_Updater_v2, self).__init__(iterator, optimizer)
		self.device = device
		self.no_of_classes = no_of_classes
		self.lambd1 = lambd1
		self.lambd2 = lambd2
		self.lambd3 = lambd3
		# self._optimizers['main'].target.freeze_layers()

	def update_core(self):
		example = self.get_iterator('main').next()
		assert len(example) == 1 # batchsize 1
		image, labels, metadata = self.converter(example)
		image = Variable(image)

		assert image.shape[0] == 1, "Batchsize of only 1 is allowed for now"

		if self.device >= 0:
			image.to_gpu(self.device)

		xp = get_array_module(image.data)
		to_substract = np.array((-1, 0))
		noise_classes = np.unique(labels[0]).astype(np.int32)
		target = xp.asarray([[0] * (self.no_of_classes)])
		gt_labels = np.setdiff1d(noise_classes, to_substract) - 1  # np.unique(labels[0]).astype(np.int32)[2:] - 1
		target[0][gt_labels] = 1

		gcam, cl_scores, class_id = self._optimizers['main'].target.stream_cl(image, gt_labels)
		# cl_scores = self._optimizers['main'].target.classify(image, is_training=True)
		interest_mask = xp.asarray([[0] * (self.no_of_classes)] * cl_scores.shape[0])
		# ignore objects we don't care about
		for obj_id in [5, 9, 11, 15, 18, 20, 21]:
			interest_mask[..., obj_id - 1] = True

		mask = self._optimizers['main'].target.get_mask(gcam)
		masked_image = self._optimizers['main'].target.mask_image(image, mask)
		masked_output = self._optimizers['main'].target.stream_am(masked_image)
		masked_output = F.sigmoid(masked_output)
		cl_loss = F.sigmoid_cross_entropy(cl_scores, target, normalize=True)
		# print("argmax class id {}, conf {}. Hand conf {} for hand={}"
		# 	  .format(class_id, cl_scores[0][class_id], cl_scores[0][-1], metadata[0]['has_hand']))
		am_loss = masked_output[0][class_id][0] # [0] if use labels for cl_stream

		# img = np.uint8(chainer_img_to_np_img(cp.asnumpy(masked_image[0].data)))
		# print("masked conf stream am: " + str(masked_output))
		# print("new conf {} am loss {}".format(masked_output[0][class_id][0], am_loss))
		# cv2.imshow('masked img', img)
		# cv2.waitKey(0)

		labels = Variable(labels)
		if self.device >= 0:
			labels.to_gpu(self.device)
		# segment_loss = self._optimizers['main'].target(image, labels)
		# + self.lambd3*segment_loss
		total_loss = self.lambd1 * cl_loss + self.lambd2 * am_loss
		report({'AM_Loss': am_loss}, self.get_optimizer('main').target)
		report({'CL_Loss': cl_loss}, self.get_optimizer('main').target)
		# report({'SG_Loss': segment_loss}, self.get_optimizer('main').target)
		report({'TotalLoss': total_loss}, self.get_optimizer('main').target)
		self._optimizers['main'].target.cleargrads()
		total_loss.backward()
		self._optimizers['main'].update()


class VOC_SEG_Updater_v2(StandardUpdater):

	def __init__(self, iterator, optimizer, no_of_classes=21, device=-1, lambd1=1.5, lambd2=1, lambd3=1.5):
		super(VOC_SEG_Updater_v2, self).__init__(iterator, optimizer)
		self.device = device
		self.no_of_classes = no_of_classes
		self.lambd1 = lambd1
		self.lambd2 = lambd2
		self.lambd3 = lambd3
		self._optimizers['main'].target.freeze_layers()

	def update_core(self):
		example = self.get_iterator('main').next()
		assert len(example) == 1  # batchsize 1
		image, labels, metadata = self.converter(example)
		image = Variable(image)

		assert image.shape[0] == 1, "Batchsize of only 1 is allowed for now"

		if self.device >= 0:
			image.to_gpu(self.device)

		xp = get_array_module(image.data)

		labels = Variable(labels)
		if self.device >= 0:
			labels.to_gpu(self.device)
		segment_loss = self._optimizers['main'].target(image, labels)
		total_loss = self.lambd3 * segment_loss
		report({'SG_Loss': segment_loss}, self.get_optimizer('main').target)
		report({'TotalLoss': total_loss}, self.get_optimizer('main').target)
		self._optimizers['main'].target.cleargrads()
		total_loss.backward()
		self._optimizers['main'].update()
