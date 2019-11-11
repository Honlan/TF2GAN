# -*- coding: utf-8 -*-

import tensorflow as tf
from glob import glob
import os

class Dataloader(object):
	def __init__(self, args):
		self.img_paths = glob(os.path.join('dataset', args.dataset_name, '*'))
		self.dataset_size = len(self.img_paths)
		self.img_size = args.img_size
		self.aug_size = int(self.img_size * 1.1)
		self.img_nc = args.img_nc
		self.batch_size = args.batch_size

		AUTOTUNE = tf.data.experimental.AUTOTUNE
		loader = tf.data.Dataset.from_tensor_slices(self.img_paths).map(self.read_image, num_parallel_calls=AUTOTUNE).cache()
		loader = loader.shuffle(self.dataset_size).repeat().map(self.augmentation, num_parallel_calls=AUTOTUNE)
		self.loader = loader.batch(self.batch_size).prefetch(AUTOTUNE)

	def read_image(self, path):
		img_str = tf.io.read_file(path)
		img = tf.cond(tf.image.is_jpeg(img_str), 
			lambda: tf.image.decode_jpeg(img_str, channels=self.img_nc), 
			lambda: tf.image.decode_png(img_str, channels=self.img_nc))
		square_size = tf.reduce_mean(tf.shape(img)[:-1])
		img = tf.image.resize_with_crop_or_pad(img, square_size, square_size)
		return img

	def augmentation(self, img):
		img = tf.image.resize(img, (self.aug_size, self.aug_size))
		img = tf.image.random_crop(img, (self.img_size, self.img_size, self.img_nc))
		img = tf.image.random_flip_left_right(img)
		return img / 127.5 - 1.