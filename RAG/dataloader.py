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
		self.tfrecord_path = os.path.join('dataset', f'{args.dataset_name}.tfrec')
		self.shorter_size = 178 if args.dataset_name == 'celeba' else args.img_size

		if args.phase == 'tfrecord':
			img_paths = glob(os.path.join('dataset', args.dataset_name, '*'))
			labels = {}
			with open(os.path.join('dataset', f'list_attr_{args.dataset_name}.txt'), 'r') as fr:
				lines = fr.readlines()
				all_attrs = lines[1].strip('\n').split()
				for i in range(2, len(lines)):
					line = lines[i].strip('\n').split()
					labels[line[0]] = {all_attrs[j]: 1.0 if int(line[j + 1]) == 1 else 0.0 for j in range(len(all_attrs))}

			labels = [[labels[img_path.split(os.sep)[-1]][attr] for attr in args.attrs] for img_path in img_paths]

			with tf.io.TFRecordWriter(self.tfrecord_path) as writer:
				for img_path, label in zip(img_paths, labels):
					feature = {
						'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[open(img_path, 'rb').read()])),
						'label': tf.train.Feature(float_list=tf.train.FloatList(value=label))
					}
					writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())

		elif args.phase == 'train':
			AUTOTUNE = tf.data.experimental.AUTOTUNE
			dataset = tf.data.TFRecordDataset(self.tfrecord_path)
			self.dataset_size = dataset.reduce(np.int64(0), lambda x, _: x + 1)
			self.desc = {
				'img': tf.io.FixedLenFeature([], 'string'), 
				'label': tf.io.FixedLenFeature([args.label_nc], 'float32')
			}
			self.loader = dataset.shuffle(min(self.dataset_size, 10000)).repeat().map(
				self.parse_example, AUTOTUNE).batch(self.batch_size).prefetch(AUTOTUNE)

		elif args.phase == 'test':
			img = self.load_image(tf.io.read_file(os.path.join('dataset', args.dataset_name, args.test_img)))
			self.img = tf.expand_dims(tf.image.resize(img, (self.img_size, self.img_size)), 0)

	def parse_example(self, example):
		feature = tf.io.parse_single_example(example, self.desc)
		return self.augmentation(self.load_image(feature['img'])), feature['label']

	def load_image(self, img_str):
		img = tf.cond(tf.image.is_jpeg(img_str), 
			lambda: tf.image.decode_jpeg(img_str, self.img_nc), 
			lambda: tf.image.decode_png(img_str, self.img_nc))
		img = tf.image.resize_with_crop_or_pad(img, self.shorter_size, self.shorter_size)
		return tf.cast(img, 'float32') / 127.5 - 1.

	def augmentation(self, img):
		img = tf.image.resize(img, (self.aug_size, self.aug_size))
		img = tf.image.random_crop(img, (self.img_size, self.img_size, self.img_nc))
		return tf.image.random_flip_left_right(img)