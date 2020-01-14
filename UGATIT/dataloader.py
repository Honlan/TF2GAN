# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from glob import glob
import os

class Dataloader(object):
	def __init__(self, args):
		self.img_size = args.img_size
		self.aug_size = int(self.img_size * 1.1)
		self.img_nc = args.img_nc
		self.batch_size = args.batch_size
		self.tfrecord_path_A = os.path.join('dataset', f'{args.dataset_name}_A.tfrec')
		self.tfrecord_path_B = os.path.join('dataset', f'{args.dataset_name}_B.tfrec')

		if args.phase == 'tfrecord':
			A_paths = glob(os.path.join('dataset', args.dataset_name, 'trainA', '*'))
			B_paths = glob(os.path.join('dataset', args.dataset_name, 'trainB', '*'))

			with tf.io.TFRecordWriter(self.tfrecord_path_A) as writer:
				for img_path in A_paths:
					feature = {'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[open(img_path, 'rb').read()]))}
					writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())

			with tf.io.TFRecordWriter(self.tfrecord_path_B) as writer:
				for img_path in B_paths:
					feature = {'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[open(img_path, 'rb').read()]))}
					writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())

		elif args.phase == 'train':
			AUTOTUNE = tf.data.experimental.AUTOTUNE
			dataset_A = tf.data.TFRecordDataset(self.tfrecord_path_A)
			dataset_B = tf.data.TFRecordDataset(self.tfrecord_path_B)
			dataset_size = dataset_A.reduce(np.int64(0), lambda x, _: x + 1)
			self.desc = {'img': tf.io.FixedLenFeature([], 'string')}
			self.loader_A = dataset_A.shuffle(min(dataset_size, 10000)).repeat().map(
				self.parse_example, AUTOTUNE).batch(self.batch_size).prefetch(AUTOTUNE)
			self.loader_B = dataset_B.shuffle(min(dataset_size, 10000)).repeat().map(
				self.parse_example, AUTOTUNE).batch(self.batch_size).prefetch(AUTOTUNE)

		elif args.phase == 'test':
			A_dataset = tf.data.Dataset.list_files(os.path.join('dataset', args.dataset_name, 'testA', '*'), False)
			B_dataset = tf.data.Dataset.list_files(os.path.join('dataset', args.dataset_name, 'testB', '*'), False)
			self.A_loader = A_dataset.map(self.load_test_image)
			self.B_loader = B_dataset.map(self.load_test_image)

	def parse_example(self, example):
		feature = tf.io.parse_single_example(example, self.desc)
		return self.augmentation(self.load_image(feature['img']))

	def load_image(self, img_str):
		img = tf.cond(tf.image.is_jpeg(img_str), 
			lambda: tf.image.decode_jpeg(img_str, self.img_nc), 
			lambda: tf.image.decode_png(img_str, self.img_nc))
		return tf.cast(img, 'float32') / 127.5 - 1.

	def augmentation(self, img):
		img = tf.image.resize(img, (self.aug_size, self.aug_size))
		img = tf.image.random_crop(img, (self.img_size, self.img_size, self.img_nc))
		return tf.image.random_flip_left_right(img)

	def load_test_image(self, img_path):
		img = self.load_image(tf.io.read_file(img_path))
		return tf.image.resize(img, (self.img_size, self.img_size))