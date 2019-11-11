# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as tk
import time
from dataloader import Dataloader
import sys
sys.path.append('..')
from ops import *
from utils import *

class Model(tk.Model):
	def __init__(self, args):
		super().__init__()
		self.args = args

	def generator(self):
		return tk.Sequential([
			Input((self.args.z_dim,)),
			Dense(4 * 4 * 512), Reshape((4, 4, 512)), BN(), Relu(),
			Deconv2d(256, 5), BN(), Relu(),
			Deconv2d(128, 5), BN(), Relu(),
			Deconv2d(64,  5), BN(), Relu(),
			Deconv2d(self.args.img_nc, 5, activation='tanh')])

	def discriminator(self):
		return tk.Sequential([
			Input((self.args.img_size, self.args.img_size, self.args.img_nc,)),
			Conv2d(64, 5, 2), Lrelu(),
			Conv2d(128, 5, 2), BN(), Lrelu(),
			Conv2d(256, 5, 2), BN(), Lrelu(),
			Conv2d(512, 5, 2), BN(), Lrelu(),
			Flatten(), Dense(1)])

	def build_model(self):
		if self.args.phase == 'train':
			self.iter = iter(Dataloader(self.args).loader)

			self.G = self.generator()
			self.D = self.discriminator()

			self.optimizer_g = tk.optimizers.Adam(learning_rate=self.args.lr, beta_1=0.5)
			self.optimizer_d = tk.optimizers.Adam(learning_rate=self.args.lr, beta_1=0.5)

			self.summary_writer = tf.summary.create_file_writer(self.args.log_dir)
			self.seed = tf.random.uniform([self.args.batch_size, self.args.z_dim], -1., 1.)
		
		elif self.args.phase == 'test':
			self.load_model()

	def train(self):
		start_time = time.time()
		samples = []
		for i in range(self.args.iteration):
			batch = next(self.iter)
			noise = tf.random.uniform([self.args.batch_size, self.args.z_dim], -1., 1.)

			with tf.GradientTape() as tape_g, tf.GradientTape() as tape_d:
				fake = self.G(noise, training=True)
				d_real = self.D(batch, training=True)
				d_fake = self.D(fake, training=True)
				loss_g = generator_loss(d_fake, self.args.gan_type)
				loss_d = discriminator_loss(d_real, d_fake, self.args.gan_type)
				print('iter: [%6d/%6d] time: %.2f loss_g: %.4f, loss_d: %.4f' % (
					i, self.args.iteration, time.time() - start_time, loss_g.numpy(), loss_d.numpy()))

			self.optimizer_g.apply_gradients(zip(tape_g.gradient(loss_g, self.G.trainable_variables), self.G.trainable_variables))
			self.optimizer_d.apply_gradients(zip(tape_d.gradient(loss_d, self.D.trainable_variables), self.D.trainable_variables))
			
			if (i + 1) % self.args.log_freq == 0:
				with self.summary_writer.as_default():
					tf.summary.scalar('loss_g', loss_g, step=i)
					tf.summary.scalar('loss_d', loss_d, step=i)

			if (i + 1) % self.args.sample_freq == 0:
				sample = self.G(self.seed, training=False)
				sample = montage(imdenorm(sample.numpy()))
				samples.append(sample)
				imsave(os.path.join(self.args.sample_dir, '{:06d}.jpg'.format(i + 1)), sample)

			if (i + 1) % self.args.save_freq == 0:
				self.save_model()

		self.save_model()
		mimsave(os.path.join(self.args.sample_dir, 'sample.gif'), samples)

	def test(self):
		result = self.G(tf.random.uniform([self.args.batch_size, self.args.z_dim], -1., 1.), training=False)
		imsave(os.path.join(self.args.result_dir, 'result.jpg'), montage(imdenorm(result.numpy())))

	def load_model(self, all_module=False):
		self.G = tk.models.load_model(os.path.join(self.args.save_dir, 'G.h5'))
		
		if all_module:
			self.D = tk.models.load_model(os.path.join(self.args.save_dir, 'D.h5'))

	def save_model(self):
		self.G.save(os.path.join(self.args.save_dir, 'G.h5'))
		self.D.save(os.path.join(self.args.save_dir, 'D.h5'))