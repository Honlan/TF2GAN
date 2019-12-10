# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os, pickle
import sys
sys.path.append('..')
from utils import array_to_list

class Dataloader(object):
	def __init__(self, args):
		self.dataset_dir = os.path.join('dataset', args.dataset_name)
		self.img_h, self.img_w, self.img_nc = args.img_h, args.img_w, args.img_nc
		self.batch_size = args.batch_size
		self.n_kps = args.n_kps
		self.n_body_part = args.n_body_part
		self.tfrecord_path = os.path.join('dataset', f'{args.dataset_name}.tfrec')

		if args.phase == 'tfrecord':
			with open(os.path.join(self.dataset_dir, args.train_pairs_csv), 'r') as fr:
				train_pairs = [line.strip('\n').split(',') for line in fr.readlines()[1:]]

			with open(os.path.join(self.dataset_dir, args.kps_bbox_pkl), 'rb') as fr:
				annos = pickle.load(fr)
				A_kps = [array_to_list(annos['train'][pair[0]]['kps']) for pair in self.train_pairs]
				B_kps = [array_to_list(annos['train'][pair[1]]['kps']) for pair in self.train_pairs]
				A_bbox = [array_to_list(annos['train'][pair[0]]['bbox']) for pair in self.train_pairs]
				B_bbox = [array_to_list(annos['train'][pair[1]]['bbox']) for pair in self.train_pairs]

			with tf.io.TFRecordWriter(self.tfrecord_path) as writer:
				for i, pair in enumerate(self.train_pairs):
					A_img = open(os.path.join(self.dataset_dir, 'train', pair[0]), 'rb').read()
					B_img = open(os.path.join(self.dataset_dir, 'train', pair[1]), 'rb').read()
					feature = {
						'A_img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[A_img])),
						'B_img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[B_img])),
						'A_kps': tf.train.Feature(float_list=tf.train.FloatList(value=A_kps[i])),
						'B_kps': tf.train.Feature(float_list=tf.train.FloatList(value=B_kps[i])),
						'A_bbox': tf.train.Feature(float_list=tf.train.FloatList(value=A_bbox[i])),
						'B_bbox': tf.train.Feature(float_list=tf.train.FloatList(value=B_bbox[i]))
					}
					writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())
		
		elif args.phase == 'train':
			AUTOTUNE = tf.data.experimental.AUTOTUNE
			dataset = tf.data.TFRecordDataset(self.tfrecord_path)
			dataset_size = dataset.reduce(np.int64(0), lambda x, _: x + 1)
			self.desc = {
				'A_img': tf.io.FixedLenFeature([], 'string'),
				'B_img': tf.io.FixedLenFeature([], 'string'),
				'A_kps': tf.io.FixedLenFeature([self.n_kps, 2], 'float32'),
				'B_kps': tf.io.FixedLenFeature([self.n_kps, 2], 'float32'),
				'A_bbox': tf.io.FixedLenFeature([self.n_body_part, 4], 'float32'),
				'B_bbox': tf.io.FixedLenFeature([self.n_body_part, 4], 'float32'),
			}
			self.loader = dataset.shuffle(min(dataset_size, 10000)).repeat().map(
				self.parse_example, AUTOTUNE).batch(self.batch_size).prefetch(AUTOTUNE)

		elif args.phase == 'test':
			with open(os.path.join(self.dataset_dir, args.test_pairs_csv), 'r') as fr:
				test_pairs = [line.strip('\n').split(',') for line in fr.readlines()[1:]]

	def parse_example(self, example):
		feature = tf.io.parse_single_example(example, self.desc)
		A_img, B_img = self.load_image(feature['A_img']), self.load_image(feature['B_img'])
		A_kps = tf.py_function(self.kps_to_htmap, [feature['A_kps']], 'float32')
		B_kps = tf.py_function(self.kps_to_htmap, [feature['B_kps']], 'float32')
		A_bbox, B_bbox = feature['A_bbox'], feature['B_bbox']
		return A_img, B_img, A_kps, B_kps, A_bbox, B_bbox

	def load_image(self, img_str):
		img = tf.cond(tf.image.is_jpeg(img_str), 
			lambda: tf.image.decode_jpeg(img_str, self.img_nc), 
			lambda: tf.image.decode_png(img_str, self.img_nc))
		return tf.cast(img, 'float32') / 127.5 - 1.

	def kps_to_htmap(self, kps):
		htmap = []
		for i in range(self.n_kps):
			if kps[i][0] == -1 or kps[i][1] == -1:
				htmap.append(tf.zeros((self.img_h, self.img_w), 'float32'))
			else:
				x, y = tf.meshgrid(tf.range(self.img_w), tf.range(self.img_h))
				x, y = tf.cast(x, 'float32'), tf.cast(y, 'float32')
				htmap.append(tf.exp(-((y - kps[i][0]) ** 2 + (x - kps[i][1]) ** 2) / (2 * 6 ** 2)))
				
		return tf.stack(htmap, -1)