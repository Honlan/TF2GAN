# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as tk
import time
from dataloader import Dataloader
import sys
sys.path.append('..')
from ops import *
from utils import *

class Model(object):
	def __init__(self, args):
		self.args = args

	def generator(self):
		x = Input((self.args.img_size, self.args.img_size, self.args.img_nc))
		c = Input((self.args.label_nc))
		
		c_tiled = tf.tile(tf.reshape(c, [-1, 1, 1, self.args.label_nc]), [1, self.args.img_size, self.args.img_size, 1])
		h = tf.concat([x, c_tiled], -1)

		h = tk.Sequential([
			Conv2d(64,  7, 1), IN(), Relu(),
			Conv2d(128, 4, 2), IN(), Relu(),
			Conv2d(256, 4, 2), IN(), Relu()])(h)

		for i in range(6):
			h = Resblock(256)(h)

		h = tk.Sequential([
			Deconv2d(128, 4), IN(), Relu(),
			Deconv2d(64,  4), IN(), Relu()])(h)

		t = Conv2d(self.args.img_nc, 7, 1, activation='tanh')(h)
		a = Conv2d(1, 7, 1, activation='sigmoid')(h)
		g = a * t + (1 - a) * x

		return tk.Model([x, c], [a, g])

	def discriminator(self):
		x = Input((self.args.img_size, self.args.img_size, self.args.img_nc))
		h = x

		dis_layer = 6
		filters = 64
		for i in range(dis_layer):
			h = lrelu(Conv2d(filters, 4, 2)(h))
			filters *= 2

		logit = Conv2d(1, 3, 1)(h)
		c = tf.reshape(Conv2d(self.args.label_nc, self.args.img_size // 2 ** dis_layer, 1, padding='valid')(h), [-1, self.args.label_nc])

		return tk.Model(x, [logit, c])

	def build_model(self):
		if self.args.phase == 'train':
			dataloader = Dataloader(self.args)
			self.args.iteration = dataloader.dataset_size // self.args.batch_size 
			self.args.attrs = dataloader.attrs
			self.args.label_nc = dataloader.label_nc
			self.iter = iter(dataloader.loader)

			self.G = self.generator()
			self.D = self.discriminator()

			boundaries = []
			values = [self.args.lr]
			for i in range(self.args.decay_epochs):
				boundaries.append(self.args.iteration * (self.args.epochs + i))
				values.append(self.args.lr - i * self.args.lr / self.args.decay_epochs)
			
			lr = tk.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
			self.optimizer_g = tk.optimizers.Adam(learning_rate=self.args.lr, beta_1=0.5, beta_2=0.999)
			self.optimizer_d = tk.optimizers.Adam(learning_rate=self.args.lr, beta_1=0.5, beta_2=0.999)

			self.summary_writer = tf.summary.create_file_writer(self.args.log_dir)
		
		elif self.args.phase == 'test':
			self.load_model()

	@tf.function
	def train_one_step(self, img, label, label_):
		with tf.GradientTape() as tape_g, tf.GradientTape() as tape_d:
			a_fake, fake = self.G([img, label_ - label])
			a_cyc, cyc = self.G([fake, label - label_])
			_, rec = self.G([img, tf.zeros_like(label)])

			d_real, c_real = self.D(img)
			d_fake, c_fake = self.D(fake)

			loss_d_adv = self.args.w_adv * discriminator_loss(d_real, d_fake, self.args.gan_type)
			loss_g_adv = self.args.w_adv * generator_loss(d_fake, self.args.gan_type)

			loss_d_cls = self.args.w_cls * c_loss(label, c_real)
			loss_g_cls = self.args.w_cls * c_loss(label_, c_fake)

			loss_g_cyc = self.args.w_cyc * l1_loss(img, cyc)
			loss_g_rec = self.args.w_rec * l1_loss(img, rec)

			loss_g_a = self.args.w_a * (l1_loss(a_fake, 0) + l1_loss(a_cyc, 0))
			loss_g_tv = self.args.w_tv * (tv_loss(a_fake) + tv_loss(a_cyc))

			loss_d = loss_d_adv + loss_d_cls
			loss_g = loss_g_adv + loss_g_cls + loss_g_cyc + loss_g_rec + loss_g_a + loss_g_tv

		self.vars_g = self.G.trainable_variables
		self.vars_d = self.D.trainable_variables
		self.optimizer_d.apply_gradients(zip(tape_d.gradient(loss_d, self.vars_d), self.vars_d))
		self.optimizer_g.apply_gradients(zip(tape_g.gradient(loss_g, self.vars_g), self.vars_g))

		item = {'loss_g_adv': loss_g_adv, 'loss_g_cls': loss_g_cls, 'loss_g_cyc': loss_g_cyc, 'loss_g_rec': loss_g_rec,
				'loss_g_a': loss_g_a, 'loss_g_tv': loss_g_tv, 'loss_d_adv': loss_d_adv, 'loss_d_cls': loss_d_cls}
		
		return item

	def train(self):
		start_time = time.time()
		step = 0
		img_size = self.args.img_size
		for e in range(self.args.epochs + self.args.decay_epochs):
			for i in range(self.args.iteration):
				img, label = next(self.iter)
				label_ = tf.random.shuffle(label)

				item = self.train_one_step(img, label, label_)
				print('epoch: [%3d/%3d] iter: [%6d/%6d] time: %.2f' % (e, self.args.epochs + self.args.decay_epochs, 
					i, self.args.iteration, time.time() - start_time))
				step += 1

				if step % self.args.log_freq == 0:
					with self.summary_writer.as_default():
						tf.summary.scalar('loss_g_adv', item['loss_g_adv'], step=step)
						tf.summary.scalar('loss_g_cls', item['loss_g_cls'], step=step)
						tf.summary.scalar('loss_g_cyc', item['loss_g_cyc'], step=step)
						tf.summary.scalar('loss_g_rec', item['loss_g_rec'], step=step)
						tf.summary.scalar('loss_g_a',   item['loss_g_a'], step=step)
						tf.summary.scalar('loss_g_tv',  item['loss_g_tv'], step=step)
						tf.summary.scalar('loss_d_adv', item['loss_d_adv'], step=step)
						tf.summary.scalar('loss_d_cls', item['loss_d_cls'], step=step)

			sample = np.zeros((2 * self.args.batch_size * img_size, (self.args.label_nc + 1) * img_size, self.args.img_nc))
			for j in range(self.args.label_nc):
				label_ = label.numpy()
				label_[:, j] = 1. - label_[:, j]

				a_fake, fake = self.G([img, label_- label])
				fake = imdenorm(fake.numpy())
				a_fake = a_fake.numpy()
				for i in range(self.args.batch_size):
					sample[2 * i * img_size: (2 * i + 1) * img_size, (j + 1) * img_size: (j + 2) * img_size] = fake[i]
					sample[(2 * i + 1) * img_size: (2 * i + 2) * img_size, (j + 1) * img_size: (j + 2) * img_size] = a_fake[i]

			img = imdenorm(img.numpy())
			for i in range(self.args.batch_size):
				sample[2 * i * img_size: (2 * i + 1) * img_size, :img_size] = img[i]

			imsave(os.path.join(self.args.sample_dir, f'{e}.jpg'), sample)
			self.save_model()

	def test(self):
		result = self.G(tf.random.uniform([self.args.batch_size, self.args.z_dim], -1., 1.), training=False)
		imsave(os.path.join(self.args.result_dir, 'result.jpg'), montage(imdenorm(result.numpy())))

	def load_model(self, all_module=False):
		self.G = self.generator()
		self.G.load_weights(os.path.join(self.args.save_dir, 'G.h5'))
		
		if all_module:
			self.D = self.discriminator()
			self.D.load_weights(os.path.join(self.args.save_dir, 'D.h5'))

	def save_model(self):
		self.G.save_weights(os.path.join(self.args.save_dir, 'G.h5'))
		self.D.save_weights(os.path.join(self.args.save_dir, 'D.h5'))
