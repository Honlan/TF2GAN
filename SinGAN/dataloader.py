# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
sys.path.append('..')
from utils import *

class Dataloader(object):
	def __init__(self, args):
		self.img = imread(os.path.join('input', 'image', args.input_image))
		self.max_size = args.max_size
		self.min_size = args.min_size
		self.scale_factor = args.scale_factor

	def get_multi_scale(self, h, w):
		if max(h, w) > self.max_size:
			sf = max(h, w) / self.max_size
			if h > w:
				h, w = self.max_size, int(w / sf)
			else:
				h, w = int(h / sf), self.max_size

		num_scale = floor(np.log(self.min_size / min(h, w)) / np.log(self.scale_factor)) + 1
		scale_factor = np.power(self.min_size / min(h, w), 1 / (num_scale - 1))
		
		sizes = [], []
		for i in range(num_scale):
			hs = int(h * np.power(scale_factor, num_scale - 1 - i))
			ws = int(w * np.power(scale_factor, num_scale - 1 - i))
			sizes.append([hs, ws])

		return num_scale, sizes

	def get_multi_scale_imgs_and_sizes(self):
		h, w = self.img.shape[:2]
		num_scale, sizes = self.get_multi_scale(h, w)
		
		imgs = []
		for i in range(num_scale):
			hs, ws = sizes[i]
			imgs.append(np.expand_dims(imnorm(imresize(self.img, hs, ws)), 0))

		return num_scale, imgs, sizes