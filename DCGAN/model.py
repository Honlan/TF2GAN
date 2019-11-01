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
			Dense(4 * 4 * 512), Reshape((4, 4, 512)), BN(), Relu(), # 4, 4, 512
			Deconv2d(256, 5), BN(), Relu(), # 8, 8, 256
			Deconv2d(128, 5), BN(), Relu(), # 16, 16, 128
			Deconv2d(64,  5), BN(), Relu(), # 32, 32, 64
			Deconv2d(self.args.img_nc, 5, activation='tanh')]) # 64, 64, self.img_nc

	def discriminator(self):
		return tk.Sequential([
			Conv2d(64, 5, 2), Lrelu(), # 32, 32, 64 
			Conv2d(128, 5, 2), BN(), Lrelu(), # 16, 16, 128
			Conv2d(256, 5, 2), BN(), Lrelu(), # 8, 8, 256
			Conv2d(512, 5, 2), BN(), Lrelu(), # 4, 4, 512
			Flatten(), Dense(1)
			])

	def build_model(self):
		self.iter = iter(Dataloader(self.args).loader)

		self.G = self.generator()
		self.D = self.discriminator()

		self.optimizer_g = tk.optimizers.Adam(learning_rate=self.args.lr, beta_1=0.5)
		self.optimizer_d = tk.optimizers.Adam(learning_rate=self.args.lr, beta_1=0.5)

		self.summary_writer = tf.summary.create_file_writer(self.args.log_dir)
		self.seed = tf.random.uniform([self.args.batch_size, self.args.z_dim], -1., 1.)

		self.G_dir = os.path.join(self.args.checkpoint_dir, 'G')
		self.D_dir = os.path.join(self.args.checkpoint_dir, 'D')
		check_dir(self.G_dir)
		check_dir(self.D_dir)

	def train(self):
		start_time = time.time()
		for i in range(self.args.iteration):
			batch = next(self.iter)
			noise = tf.random.uniform([self.args.batch_size, self.args.z_dim], -1., 1.)

			with tf.GradientTape() as tape_g, tf.GradientTape() as tape_d:
				fake = self.G(noise, training=True)
				d_real = self.D(batch, training=True)
				d_fake = self.D(fake, training=True)
				loss_g = generator_loss(d_fake, self.args.gan_type)
				loss_d = discriminator_loss(d_real, d_fake, self.args.gan_type)
				print('iter: [%6d/%6d] time: %4.4f loss_g: %.6f, loss_d: %.6f' % (i, self.args.iteration, time.time() - start_time, loss_g.numpy(), loss_d.numpy()))

			self.optimizer_g.apply_gradients(zip(tape_g.gradient(loss_g, self.G.trainable_variables), self.G.trainable_variables))
			self.optimizer_d.apply_gradients(zip(tape_d.gradient(loss_d, self.D.trainable_variables), self.D.trainable_variables))
			
			if (i + 1) % self.args.log_freq == 0:
				with self.summary_writer.as_default():
					tf.summary.scalar('loss_g', loss_g, step=i)
					tf.summary.scalar('loss_d', loss_d, step=i)

			if (i + 1) % self.args.sample_freq == 0:
				sample = self.G(self.seed, training=False)
				imsave(os.path.join(self.args.sample_dir, '{:06d}.jpg'.format(i + 1)), montage(imdenorm(sample.numpy())))

			if (i + 1) % self.args.checkpoint_freq == 0:
				self.save_model()

		self.save_model()

	def test(self):
		self.load_model(all_module=False)
		result = self.G(tf.random.uniform([self.args.batch_size, self.args.z_dim], -1., 1.), training=False)
		imsave(os.path.join(self.args.result_dir, 'result.jpg'), montage(imdenorm(result.numpy())))

	def load_model(self, all_module=True):
		self.G = tf.saved_model.load(self.G_dir)
		
		if all_module:
			self.D = tf.saved_model.load(self.D_dir)

	def save_model(self):
		tf.saved_model.save(self.G, self.G_dir)
		tf.saved_model.save(self.D, self.D_dir)