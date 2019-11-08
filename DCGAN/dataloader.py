# -*- coding: utf-8 -*-

import tensorflow as tf
from glob import glob
import os

class Dataloader(object):
	def __init__(self, args):
		self.img_paths = glob(os.path.join('dataset', args.dataset_name, '*'))
		self.data_num = len(self.img_paths)
		self.img_size = args.img_size
		self.img_nc = args.img_nc
		self.batch_size = args.batch_size

		AUTOTUNE = tf.data.experimental.AUTOTUNE
		loader = tf.data.Dataset.from_tensor_slices(self.img_paths).shuffle(self.data_num).repeat()
		self.loader = loader.map(self.preprocess, num_parallel_calls=AUTOTUNE).batch(self.batch_size).prefetch(AUTOTUNE)

	def preprocess(self, img_path, aug=True):
		img_str = tf.io.read_file(img_path)
		img = tf.cond(tf.image.is_jpeg(img_str), lambda: tf.image.decode_jpeg(img_str, channels=3), lambda: tf.image.decode_png(img_str, channels=3))
		square_size = tf.reduce_mean(tf.shape(img)[:-1])
		img = tf.image.resize_with_crop_or_pad(img, square_size, square_size)
		
		if aug:
			aug_size = int(self.img_size * 1.1)
			img = tf.image.random_flip_left_right(tf.image.random_crop(tf.image.resize(img, (aug_size, aug_size)), (self.img_size, self.img_size, self.img_nc)))
		else:
			img = tf.image.resize(img, (self.img_size, self.img_size))

		return img / 127.5 - 1.