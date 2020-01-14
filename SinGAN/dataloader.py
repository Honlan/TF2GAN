# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
sys.path.append('..')
from utils import *

class Dataloader(object):
	def __init__(self, args):
		self.img = imread(os.path.join('asset', args.input_image))[:, :, :args.img_nc]
		self.max_size = args.max_size
		self.min_size = args.min_size
		self.scale_factor = args.scale_factor

	def get_multi_scale_sizes(self, h, w):
		if max(h, w) > self.max_size:
			sf = max(h, w) / self.max_size
			if h > w:
				h, w = self.max_size, int(w / sf)
			else:
				h, w = int(h / sf), self.max_size
	
		sizes = [[h, w]]
		
		while True:
			h = ceil(h * self.scale_factor)
			w = ceil(w * self.scale_factor)
			
			if min(h, w) > self.min_size:
				sizes.append([h, w])
			else:
				break

		return len(sizes), sizes[::-1]

	def get_multi_scale_imgs_and_sizes(self):
		h, w = self.img.shape[:2]
		num_scale, sizes = self.get_multi_scale_sizes(h, w)
		
		imgs = []
		for i in range(num_scale):
			hs, ws = sizes[i]
			imgs.append(np.expand_dims(imnorm(imresize(self.img, hs, ws)), 0))

		return num_scale, imgs, sizes