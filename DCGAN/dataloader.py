# -*- coding: utf-8 -*-

import tensorflow as tf
from glob import glob
import os, time
import sys
sys.path.append('..')
from utils import imread

class Dataloader(object):
	def __init__(self, args, load_data_to_memory=True):
		self.imgs = glob(os.path.join('dataset', args.dataset_name, '*'))
		self.dataset_size = len(self.imgs)
		self.img_size = args.img_size
		self.img_nc = args.img_nc
		self.batch_size = args.batch_size
		self.load_data_to_memory = load_data_to_memory

		if self.load_data_to_memory:
			print('Loading data to memory...')
			t0 = time.time()
			self.imgs = [imread(path, norm=False) for path in self.imgs]
			print('Load finished in %.2fs...' % (time.time() - t0))

		AUTOTUNE = tf.data.experimental.AUTOTUNE
		loader = tf.data.Dataset.from_tensor_slices(self.imgs).map(self.preprocess, num_parallel_calls=AUTOTUNE)
		self.loader = loader.shuffle(self.dataset_size).repeat().batch(self.batch_size).prefetch(AUTOTUNE)

	def preprocess(self, img, aug=True):
		if not self.load_data_to_memory:
			img_str = tf.io.read_file(img)
			img = tf.cond(tf.image.is_jpeg(img_str), lambda: tf.image.decode_jpeg(img_str, channels=3), lambda: tf.image.decode_png(img_str, channels=3))
		
		square_size = tf.reduce_mean(tf.shape(img)[:-1])
		img = tf.image.resize_with_crop_or_pad(img, square_size, square_size)
		
		if aug:
			aug_size = int(self.img_size * 1.1)
			img = tf.image.random_flip_left_right(tf.image.random_crop(tf.image.resize(img, (aug_size, aug_size)), (self.img_size, self.img_size, self.img_nc)))
		else:
			img = tf.image.resize(img, (self.img_size, self.img_size))

		return img / 127.5 - 1.