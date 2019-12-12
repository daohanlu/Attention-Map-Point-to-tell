import cv2
from chainer.training import StandardUpdater
from chainer import Variable
from chainer import report
from chainer import functions as F
from chainer.backends.cuda import get_array_module
import numpy as np


class VOC_ClassificationUpdater(StandardUpdater):
	def __init__(self, iterator, optimizer, no_of_classes=20, device=-1):
		super(VOC_ClassificationUpdater, self).__init__(iterator, optimizer)
		self.device = device
		self.no_of_classes=no_of_classes

		self._optimizers['main'].target.freeze_layers()
	def update_core(self):

		image, labels, metadata = self.converter(self.get_iterator('main').next())
		assert image.shape[0] == 1, "Batchsize of only 1 is allowed for now"
		image = Variable(image)

		if self.device >= 0:
			image.to_gpu(self.device)
		cl_output = self._optimizers['main'].target.classify(image)
		xp = get_array_module(cl_output.data)

		target = xp.asarray([[0]*(self.no_of_classes)]*cl_output.shape[0])
		for i in range(labels.shape[0]):
			gt_labels = np.unique(labels[i]).astype(np.int32)[2:] - 1 # Not considering -1 & 0
			target[i][gt_labels] = 1
		loss = F.sigmoid_cross_entropy(cl_output, target, normalize=True)
		report({'Loss':loss}, self.get_optimizer('main').target)
		self._optimizers['main'].target.cleargrads()
		loss.backward()
		self._optimizers['main'].update()



class VOC_GAIN_Updater(StandardUpdater):

	def __init__(self, iterator, optimizer, no_of_classes=20, device=-1, lambd1=1.5, lambd2=1, lambd3=1.5):
		super(VOC_GAIN_Updater, self).__init__(iterator, optimizer)
		self.device = device
		self.no_of_classes = no_of_classes
		self.lambd1 = lambd1
		self.lambd2 = lambd2
		self.lambd3 = lambd3

		self._optimizers['main'].target.freeze_layers()

	def update_core(self):
		image, labels, metadata = self.converter(self.get_iterator('main').next())
		# img = np.transpose(image[0], (1, 2, 0))
		# print(img)
		# cv2.imshow('img', np.uint8(img))
		# cv2.imshow('lbl', np.uint8(labels[0]))
		# cv2.waitKey(0)
		image = Variable(image)

		assert image.shape[0] == 1, "Batchsize of only 1 is allowed for now"

		if self.device >= 0:
			image.to_gpu(self.device)

		xp = get_array_module(image.data)
		to_substract = np.array((-1, 0))
		noise_classes = np.unique(labels[0]).astype(np.int32)
		# print(noise_classes)
		target = xp.asarray([[0] * (self.no_of_classes)])
		gt_labels = np.setdiff1d(noise_classes, to_substract) - 1  # np.unique(labels[0]).astype(np.int32)[2:] - 1
		target[0][gt_labels] = 1
		
		gcam, cl_scores, class_id, fconf, floc = self._optimizers['main'].target.stream_cl(image, gt_labels)

		mask = self._optimizers['main'].target.get_mask(gcam)
		masked_image = self._optimizers['main'].target.mask_image(image, mask)
		masked_output = self._optimizers['main'].target.stream_am(masked_image)
		masked_output = F.sigmoid(masked_output)

		cl_loss = F.sigmoid_cross_entropy(cl_scores, target, normalize=True)
		am_loss = masked_output[0][class_id][0]
		finger_loss = self.finger_loss(fconf, floc, metadata, xp)

		labels = Variable(labels)
		if self.device >= 0:
			labels.to_gpu(self.device)
		segment_loss = self._optimizers['main'].target(image, labels)
		total_loss = self.lambd1 * cl_loss + self.lambd2 * am_loss + self.lambd3*segment_loss + 1*finger_loss
		report({'AM_Loss': am_loss}, self.get_optimizer('main').target)
		report({'CL_Loss': cl_loss}, self.get_optimizer('main').target)
		report({'SG_Loss': segment_loss}, self.get_optimizer('main').target)
		report({'TotalLoss': total_loss}, self.get_optimizer('main').target)
		self._optimizers['main'].target.cleargrads()
		# if not np.isnan(float(total_loss.data)):
		total_loss.backward()
		self._optimizers['main'].update()

	def finger_loss(self, pred_conf, pred_loc, metadata, xp):
		if metadata['has_finger']:
			conf_gt = xp.asarray([[1, 0]])
			loc_gt = xp.asarray([[0] * 2])
			conf_loss = F.sigmoid_cross_entropy(pred_conf, conf_gt, normalize=True)
			loc_loss = F.sigmoid_cross_entropy(pred_loc, loc_gt, normalize=True)
		else:
			conf_gt = xp.asarray[[0, 1]]
			loc_gt = xp.asarray([[0] * 2])
			conf_loss = F.sigmoid_cross_entropy(pred_conf, conf_gt, normalize=True)
			loc_loss = F.sigmoid_cross_entropy(pred_loc, loc_gt, normalize=True)
		return 0.5 * conf_loss + 0.5 * loc_loss