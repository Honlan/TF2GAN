# -*- coding: utf-8 -*-

import tensorflow as tf
from glob import glob
import os

class Dataloader(object):
	def __init__(self, args):
		self.img_size = args.img_size
		self.aug_size = int(self.img_size * 1.1)
		self.img_nc = args.img_nc
		self.batch_size = args.batch_size
		self.label_nc = args.label_nc
		self.tfrecord_path = os.path.join('dataset', f'{args.dataset_name}.tfrec')

		if args.phase == 'tfrecord':
			img_paths = glob(os.path.join('dataset', args.dataset_name, 'img', '*'))
			label_paths = glob(os.path.join('dataset', args.dataset_name, 'label', '*'))
			img_paths.sort()
			label_paths.sort()

			with tf.io.TFRecordWriter(self.tfrecord_path) as writer:
				for img_path, label_path in zip(img_paths, label_paths):
					feature = {
						'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[open(img_path, 'rb').read()])),
						'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[open(label_path, 'rb').read()]))
					}
					writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())

		elif args.phase == 'train':
			AUTOTUNE = tf.data.experimental.AUTOTUNE
			dataset = tf.data.TFRecordDataset(self.tfrecord_path)
			self.dataset_size = dataset.reduce(0, lambda x, _: x + 1)
			self.desc = {'img': tf.io.FixedLenFeature([], 'string'), 'label': tf.io.FixedLenFeature([], 'string')}
			self.loader = dataset.shuffle(min(self.dataset_size, 10000)).repeat().map(
				self.parse_example, AUTOTUNE).batch(self.batch_size).prefetch(AUTOTUNE)

		elif args.phase == 'test':
			dataset = tf.data.Dataset.list_files(os.path.join('dataset', args.dataset_name, args.test_img_dir, '*'))
			self.loader = dataset.map(self.load_test_data)

	def parse_example(self, example):
		feature = tf.io.parse_single_example(example, self.desc)
		img, label = self.load_image(feature['img'], self.img_nc, True), self.load_image(feature['label'], 1, False)
		return self.augmentation(img, label)

	def load_image(self, img_str, nc, norm):
		img = tf.cond(tf.image.is_jpeg(img_str), 
			lambda: tf.image.decode_jpeg(img_str, nc), 
			lambda: tf.image.decode_png(img_str, nc))
		return tf.cast(img, 'float32') / 127.5 - 1. if norm else img

	def augmentation(self, img, label):
		img = tf.image.resize(img, (self.aug_size, self.aug_size))
		label = tf.image.resize(label, (self.aug_size, self.aug_size), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		label = tf.one_hot(label[:, :, 0], depth=self.label_nc, on_value=1., off_value=0.)

		data = tf.concat([img, label], -1)
		data = tf.image.random_crop(data, (self.img_size, self.img_size, self.img_nc + self.label_nc))
		return data[:, :, :self.img_nc], data[:, :, self.img_nc:]

	def load_test_data(self, img_path):
		img = self.load_image(tf.io.read_file(img_path), self.img_nc, True)
		return img_path, tf.image.resize(img, (self.img_size, self.img_size))