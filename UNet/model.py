# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as tk
import numpy as np
import time
from dataloader import Dataloader
import sys
sys.path.append('..')
from ops import *
from utils import *

class Model(object):
	def __init__(self, args):
		self.args = args

	def parser(self):
		x = Input((self.args.img_size, self.args.img_size, self.args.img_nc))
		layers = []
		
		h = Conv2d(64, 4, 2, use_bias=False)(x)
		layers.append(h)

		for filters in [128, 256, 512, 512]:
			h = BN()(Conv2d(filters, 4, 2, use_bias=False)(lrelu(h)))
			layers.append(h)

		h = BN()(Deconv2d(512, 4, use_bias=False)(relu(Conv2d(512, 4, 2, use_bias=False)(lrelu(h)))))

		for i, filters in enumerate([512, 256, 128, 64]):
			h = tf.concat([h, layers[-i - 1]], -1)
			h = BN()(Deconv2d(filters, 4, use_bias=False)(relu(h)))
			if i == 0:
				h = Dropout(0.5)(h)

		h = tf.concat([h, layers[0]], -1)
		logit = Deconv2d(self.args.label_nc, 4)(relu(h))

		return tk.Model(x, logit)
		
	def build_model(self):
		if self.args.phase == 'train':
			dataloader = Dataloader(self.args)
			self.args.iteration = dataloader.dataset_size // self.args.batch_size 
			self.iter = iter(dataloader.loader)

			self.P = self.parser()

			boundaries = []
			values = [self.args.lr]
			for i in range(self.args.decay_epochs):
				boundaries.append(self.args.iteration * (self.args.epochs + i))
				values.append(self.args.lr - i * self.args.lr / self.args.decay_epochs)

			lr = tk.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
			self.optimizer = tk.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)

			self.summary_writer = tf.summary.create_file_writer(self.args.log_dir)
		
		elif self.args.phase == 'test':
			self.loader = Dataloader(self.args).loader
			self.load()

	@tf.function
	def train_step(self, img, label):
		with tf.GradientTape() as tape:
			label_ = self.P(img, training=True)
			loss = c_loss(label, label_, False)
			accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(label, axis=-1), tf.argmax(label_, axis=-1)), 'float32'))

		vars_p = self.P.trainable_variables
		self.optimizer.apply_gradients(zip(tape.gradient(loss, vars_p), vars_p))

		return {'loss': loss, 'accuracy': accuracy}

	def train(self):
		start_time = time.time()
		step = 0
		img_size = self.args.img_size
		for e in range(self.args.epochs + self.args.decay_epochs):
			for i in range(self.args.iteration):
				img, label = next(self.iter)

				item = self.train_step(img, label)
				print('epoch: [%3d/%3d] iter: [%6d/%6d] time: %.2f' % (e, self.args.epochs + self.args.decay_epochs, 
					i, self.args.iteration, time.time() - start_time))
				step += 1
				
				if step % self.args.log_freq == 0:
					with self.summary_writer.as_default():
						tf.summary.scalar('loss', item['loss'], step=step)
						tf.summary.scalar('accuracy', item['accuracy'], step=step)

			label_ = self.P(img, training=False)
			img, label, label_ = imdenorm(img.numpy()), label.numpy(), label_.numpy()
			sample = np.zeros((self.args.batch_size * img_size, 3 * img_size, self.args.img_nc))

			for i in range(self.args.batch_size):
				sample[i * img_size: (i + 1) * img_size, 0 * img_size: 1 * img_size] = img[i]
				sample[i * img_size: (i + 1) * img_size, 1 * img_size: 2 * img_size] = self.multi_to_one(label[i])
				sample[i * img_size: (i + 1) * img_size, 2 * img_size: 3 * img_size] = self.multi_to_one(label_[i])

			imsave(os.path.join(self.args.sample_dir, f'{e}.jpg'), sample)
			self.save()

	def test(self):
		result_dir = os.path.join(self.args.result_dir, self.args.test_img_dir)
		check_dir(result_dir)
		for path, img in self.loader:
			filename = path.numpy().split(os.sep)[-1]
			label_ = self.P(tf.expand_dims(img, 0), training=False)
			imsave(os.path.join(result_dir, filename), self.multi_to_one(label_[0]))

	def multi_to_one(self, data):
		return np.expand_dims(np.argmax(data, -1) / (self.args.label_nc - 1), -1)

	def load(self):
		self.P = tk.models.load_model(os.path.join(self.args.save_dir, 'P.h5'))

	def save(self):
		self.P.save(os.path.join(self.args.save_dir, 'P.h5'))