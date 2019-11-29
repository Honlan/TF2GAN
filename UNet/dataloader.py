# -*- coding: utf-8 -*-

import tensorflow as tf
from glob import glob
import os, random

class Dataloader(object):
	def __init__(self, args):
		self.img_paths = glob(os.path.join('dataset', args.dataset_name, 'img', '*'))
		self.label_paths = glob(os.path.join('dataset', args.dataset_name, 'label', '*'))
		self.img_paths.sort()
		self.label_paths.sort()
		self.dataset_size = len(self.img_paths)
		self.img_size = args.img_size
		self.aug_size = int(self.img_size * 1.1)
		self.img_nc = args.img_nc
		self.batch_size = args.batch_size
		self.tfrecord_path = os.path.join('dataset', f'{args.dataset_name}.tfrec')

		if args.dataset_name == 'CelebAMask19':
			self.label_names = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear',
								'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
			self.label_nc = len(self.label_names)

		num_parallel_calls = 8
		buffer_size = 100

		if args.phase == 'tfrecord':
			with tf.io.TFRecordWriter(self.tfrecord_path) as writer:
				for img_path, label_path in zip(self.img_paths, self.label_paths):
					img, label = open(img_path, 'rb').read(), open(label_path, 'rb').read()
					feature = {
						'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
						'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
					}
					writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())

		elif args.phase == 'train':
			dataset = tf.data.TFRecordDataset(self.tfrecord_path)
			self.desc = {'img': tf.io.FixedLenFeature([], 'string'), 'label': tf.io.FixedLenFeature([], 'string')}
			self.loader = dataset.shuffle(min(self.dataset_size, 10000)).repeat().map(
				self.parse_example, num_parallel_calls).batch(self.batch_size).prefetch(buffer_size)

	def parse_example(self, example):
		feature = tf.io.parse_single_example(example, self.desc)
		img, label = self.load_image(feature['img'], self.img_nc, True), self.load_image(feature['label'], 1, False)
		img, label = self.augmentation(img, label)
		return img, label

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
		img, label = data[:, :, :self.img_nc], data[:, :, self.img_nc:]
		return img, label