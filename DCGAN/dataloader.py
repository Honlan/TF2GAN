# -*- coding: utf-8 -*-

import tensorflow as tf
from glob import glob
import os

class Dataloader(object):
	def __init__(self, args):
		self.img_paths = glob(os.path.join('datasets', args.dataset_name, '*'))
		self.data_num = len(self.img_paths)
		self.img_size = args.img_size
		self.img_nc = args.img_nc
		self.batch_size = args.batch_size

		AUTOTUNE = tf.data.experimental.AUTOTUNE
		loader = tf.data.Dataset.from_tensor_slices(self.img_paths).map(self.preprocess, num_parallel_calls=AUTOTUNE)
		self.loader = loader.apply(tf.data.experimental.shuffle_and_repeat(self.data_num)).batch(self.batch_size).prefetch(buffer_size=AUTOTUNE)

	def preprocess(self, img_path, aug=True):
		img = tf.image.central_crop(tf.image.decode_image(tf.io.read_file(img_path), channels=self.img_nc))
		
		if aug:
			aug_size = int(self.img_size * 1.1)
			img = tf.image.random_flip_left_right(tf.image.random_crop(tf.image.resize(img, aug_size), self.img_size))
		else:
			img = tf.image.resize(img, self.img_size)

		return img / 127.5 - 1.