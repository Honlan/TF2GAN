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
		self.tfrecord_path = os.path.join('dataset', f'{args.dataset_name}.tfrec')
		if args.dataset_name == 'celeba':
			self.shorter_size = 178

		num_parallel_calls = 8
		buffer_size = 100

		if args.phase == 'tfrecord':
			with tf.io.TFRecordWriter(self.tfrecord_path) as writer:
				for img_path in self.img_paths:
					img = open(img_path, 'rb').read()
					feature = {'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img]))}
					writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())

		elif args.phase == 'train':
			dataset = tf.data.TFRecordDataset(self.tfrecord_path)
			self.desc = {'img': tf.io.FixedLenFeature([], 'string')}
			self.loader = dataset.shuffle(min(self.dataset_size, 10000)).repeat().map(
				self.parse_example, num_parallel_calls).batch(self.batch_size).prefetch(buffer_size)

	def parse_example(self, example):
		feature = tf.io.parse_single_example(example, self.desc)
		img = self.load_image(feature['img'])
		img = self.augmentation(img)
		return img

	def load_image(self, img_str):
		img = tf.cond(tf.image.is_jpeg(img_str), 
			lambda: tf.image.decode_jpeg(img_str, self.img_nc), 
			lambda: tf.image.decode_png(img_str, self.img_nc))
		img = tf.image.resize_with_crop_or_pad(img, self.shorter_size, self.shorter_size)
		img = tf.cast(img, 'float32') / 127.5 - 1.
		return img

	def augmentation(self, img):
		img = tf.image.resize(img, (self.aug_size, self.aug_size))
		img = tf.image.random_crop(img, (self.img_size, self.img_size, self.img_nc))
		img = tf.image.random_flip_left_right(img)
		return img