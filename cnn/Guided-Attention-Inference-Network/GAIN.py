import chainer
import chainer.functions as F


class GAIN(chainer.Chain):
	def __init__(self):
		super(GAIN, self).__init__()
		# To override in child class or 
		# set after initiations from respective class function
		# eg see set_final_conv_layer, set_grad_target_layer
		# set_GAIN_functions
		
		self.size = None  # Size of images
		self.GAIN_functions = None  # Refer files in /models
		self.final_conv_layer = None
		self.grad_target_layer = None

	def stream_cl(self, inp, label=None, use_max_pooling=False):
		h = inp
		for key, funcs in self.GAIN_functions.items():
			for func in funcs:
				h = func(h)
			if key == self.final_conv_layer:
				activation = h
			if key == self.grad_target_layer:
				break
		gcam, class_id = self.get_gcam(h, activation, (inp.shape[-2], inp.shape[-1]), label=label, use_max_pooling=use_max_pooling)
		return gcam, h, class_id

	def stream_cl_multi(self, inp, label=None, use_max_pooling=False):
		"""
		Returns the classification given an images as well the attention map for each present object category
		"""
		h = inp
		for key, funcs in self.GAIN_functions.items():
			for func in funcs:
				h = func(h)
			if key == self.final_conv_layer:
				activation = h
			if key == self.grad_target_layer:
				break
		gcams, class_ids = self.get_gcam_multi(h, activation, (inp.shape[-2], inp.shape[-1]), use_max_pooling)
		return gcams, h, class_ids

	def stream_am(self, masked_image):
		h = masked_image
		for key, funcs in self.GAIN_functions.items():
			for func in funcs:
				h = func(h)

		return h

	def stream_ext(self, inp):
		raise NotImplementedError

	def get_gcam(self, end_output, activations, shape, label, use_max_pooling):
		self.cleargrads()
		# end_output: class confidences
		class_id = self.set_init_grad(end_output, label)
		end_output.backward(retain_grad=True)
		grad = activations.grad_var
		if use_max_pooling:
			grad = F.max_pooling_2d(grad, (grad.shape[-2], grad.shape[-1]), 1)
		else:
			grad = F.average_pooling_2d(grad, (grad.shape[-2], grad.shape[-1]), 1)
		grad = F.expand_dims(F.reshape(grad, (grad.shape[0]*grad.shape[1], grad.shape[2], grad.shape[3])), 0)
		weights = activations
		weights = F.expand_dims(F.reshape(weights, (weights.shape[0]*weights.shape[1], weights.shape[2], weights.shape[3])), 0)
		gcam = F.resize_images(F.relu(F.convolution_2d(weights, grad, None, 1, 0)), shape)
		return gcam, class_id

	def get_gcam_multi(self, end_output, activations, shape, use_max_pooling):
		gcams = []
		class_ids = []
		for i in range(len(end_output[0])):
			if end_output[0][i].data > 0:
				self.cleargrads()
				class_id = self.set_init_grad(end_output, label=[i])
				end_output.backward(retain_grad=True)
				grad = activations.grad_var
				if use_max_pooling:
					grad = F.max_pooling_2d(grad, (grad.shape[-2], grad.shape[-1]), 1)
				else:
					grad = F.average_pooling_2d(grad, (grad.shape[-2], grad.shape[-1]), 1) # this is the default, and the method specified in GAIN paper.
				grad = F.expand_dims(F.reshape(grad, (grad.shape[0]*grad.shape[1], grad.shape[2], grad.shape[3])), 0)
				weights = activations
				weights = F.expand_dims(F.reshape(weights, (weights.shape[0]*weights.shape[1], weights.shape[2], weights.shape[3])), 0)
				gcam = F.resize_images(F.relu(F.convolution_2d(weights, grad, None, 1, 0)), shape)
				gcams.append(gcam)
				class_ids.append(class_id)
		return gcams, class_ids

	def set_init_grad(self, var, label):
		# var: class confidences
		var.grad = self.xp.zeros_like(var.data)
		if label is None:
			class_id = F.argmax(var).data
			var.grad[0][class_id] = 1

		else:
			class_id = self.xp.random.choice(label, 1)
			var.grad[0][class_id] = 1
		return class_id

	def add_freeze_layers(self, links_list):
		self.freezed_layers = links_list

	def freeze_layers(self):
		for link in self.freezed_layers:
			getattr(self, link).disable_update()

	def set_final_conv_layer(self, layername):
		self.final_conv_layer = layername

	def set_grad_target_layer(self, layername):
		self.grad_target_layer = layername

	def set_GAIN_functions(self, ordered_dict):
		for key in ordered_dict.keys():
			for item_no in range(len(ordered_dict[key])):
				if isinstance(ordered_dict[key][item_no], str):
					ordered_dict[key][item_no] = getattr(self, ordered_dict[key][item_no])
		self.GAIN_functions = ordered_dict
		
	@staticmethod
	def get_mask(gcam, sigma=.5, w=8):
		gcam = (gcam - F.min(gcam).data)/(F.max(gcam) - F.min(gcam)).data
		mask = F.squeeze(F.sigmoid(w * (gcam - sigma)))
		return mask

	@staticmethod
	def mask_image(img, mask):
		broadcasted_mask = F.broadcast_to(mask, img.shape)
		to_subtract = img*broadcasted_mask
		return img - to_subtract
