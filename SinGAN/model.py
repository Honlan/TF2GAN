# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as tk
import time, pickle
from dataloader import Dataloader
import sys
sys.path.append('..')
from ops import *
from utils import *

class Model(tk.Model):
	def __init__(self, args):
		super().__init__()
		self.args = args
		self.num_scale, self.imgs, self.sizes = Dataloader(args).get_multi_scale_imgs_and_sizes()

		for i in range(self.num_scale):
			self.imgs[i] = tf.constant(self.imgs[i], tf.float32)
			self.sizes[i].append(self.args.img_nc)

	def generator(self, h, w, c, filters):
		x = Input((h, w, c,))
		y = Input((h, w, c,))

		S = tk.Sequential([x,
			Conv2d(filters, 3, 1), BN(), Lrelu(),
			Conv2d(filters, 3, 1), BN(), Lrelu(),
			Conv2d(filters, 3, 1), BN(), Lrelu(),
			Conv2d(filters, 3, 1), BN(), Lrelu(),
			Conv2d(self.args.img_nc, 3, 1, activation='tanh')])

		return tk.Model([x, y], Add()([S.output, y]))

	def discriminator(self, h, w, c, filters):
		return tk.Sequential([
			Input((h, w, c,)),
			Conv2d(filters, 3, 1), BN(), Lrelu(),
			Conv2d(filters, 3, 1), BN(), Lrelu(),
			Conv2d(filters, 3, 1), BN(), Lrelu(),
			Conv2d(filters, 3, 1), BN(), Lrelu(),
			Conv2d(1, 3, 1)])

	def build_model(self):
		if self.args.phase == 'train':
			self.Gs = []
			self.Ds = []
			self.noise_weights = []
		
		elif self.args.phase == 'test':
			self.load_model()

	def train(self):
		filters_prev = 0
		start_time = time.time()

		for scale in range(self.num_scale):
			img = self.imgs[scale]
			imsave(os.path.join(self.args.save_dir, f'real_{scale}.jpg'), imdenorm(img.numpy()[0]))

			filters = min(self.args.num_filter * np.power(2, scale // 4), 128)
			h, w, c = self.sizes[scale][0], self.sizes[scale][1], self.sizes[scale][2]

			if filters == filters_prev:
				G = tk.models.load_model(os.path.join(self.args.save_dir, f'G_{scale - 1}.h5'))
				D = tk.models.load_model(os.path.join(self.args.save_dir, f'D_{scale - 1}.h5'))
			else:
				G = self.generator(h, w, c, filters)
				D = self.discriminator(h, w, c, filters)
			
			self.Gs.append(G)
			self.Ds.append(D)

			lr = tk.optimizers.schedules.ExponentialDecay(self.args.lr, decay_steps=self.args.decay_steps, decay_rate=self.args.decay_rate)
			self.optimizer_g = tk.optimizers.Adam(learning_rate=lr, beta_1=0.5)
			self.optimizer_d = tk.optimizers.Adam(learning_rate=lr, beta_1=0.5)

			rec_prev = self.get_prev('rec')
			noise_weight = 1. if scale == 0 else self.args.noise_weight * tf.sqrt(l2_loss(img, rec_prev))
			self.noise_weights.append(noise_weight)

			for i in range(self.args.iteration):
				z = tf.random.uniform([1, h, w, c], -1., 1.)
				z_fixed = tf.random.uniform(tf.shape(self.imgs[0]), -1., 1.)

				h0, w0, c0 = self.sizes[0][0], self.sizes[0][1], self.args.img_nc
				self.

				for j in range(self.args.D_step):
					with tf.GradientTape() as tape_d:
						d_real = self.D(img, training=True)
						
						fake_prev = self.get_prev('random')
						fake = self.G([noise_weight * z + fake_prev, fake_prev], training=True)
						d_fake = self.D(fake, training=True)
						loss_d_adv = discriminator_loss(d_real, d_fake, self.args.gan_type)
						
						alpha = tf.random.uniform([1, 1, 1, 1], 0., 1.)
						inter_sample = alpha * img + (1 - alpha) * fake
						loss_d_gp = gradient_penalty(self.D, inter_sample, self.args.w_gp)

						loss_d = loss_d_adv + loss_d_gp

					self.optimizer_d.apply_gradients(zip(tape_d.gradient(loss_d, self.D.trainable_variables), self.D.trainable_variables))

				for j in range(self.args.G_step):
					with tf.GradientTape() as tape_g:
						fake_prev = self.get_prev('random')
						fake = self.G([noise_weight * z + fake_prev, fake_prev], training=True)
						d_fake = self.D(fake, training=True)
						loss_g_adv = generator_loss(d_fake, self.args.gan_type)

						zr = noise_weight * self.z_fixed if scale == 0 else rec_prev
						rec = self.G([zr, rec_prev], training=True)
						loss_g_rec = self.args.w_rec * l2_loss(rec, img)
						loss_g = loss_g_adv + loss_g_rec

					self.optimizer_g.apply_gradients(zip(tape_g.gradient(loss_g, self.G.trainable_variables), self.G.trainable_variables))

				print('scale: [%d/%d] iter: [%4d/%4d] time: %.2f loss_g_adv: %.4f, loss_g_rec: %.4f, loss_d_adv: %.4f, loss_d_gp: %.4f' % (
					scale, self.num_scale, i, self.args.iteration, time.time() - start_time, loss_g_adv.numpy(), loss_g_rec.numpy(), loss_d_adv.numpy(), loss_d_gp.numpy()))

				if (i + 1) % self.args.sample_freq == 0:
					imsave(os.path.join(self.args.save_dir, f'fake_{scale}_{i + 1}.jpg'), imdenorm(fake.numpy()[0]))
					imsave(os.path.join(self.args.save_dir, f'rec_{scale}_{i + 1}.jpg'), imdenorm(rec.numpy()[0]))

			filters_prev = filters
			self.Gs.append(self.G)
			self.save_model(scale)

	def get_prev(self, mode='random'):
		prev = 0.
		
		if mode == 'random':
			for i in range(len(self.Gs)):
				h, w, c = self.sizes[i][0], self.sizes[i][1], self.args.img_nc
				z = tf.random.uniform([1, h, w, c], -1., 1.)
				prev = self.Gs[i]([self.noise_weights[i] * z + prev, prev], training=False)
				prev = tf.image.resize(prev, (self.sizes[i + 1][0], self.sizes[i + 1][1]))
		
		elif mode == 'rec':
			for i in range(len(self.Gs)):
				zr = self.noise_weights[i] * self.z_fixed if i == 0 else prev
				prev = self.Gs[i]([zr, prev], training=False)
				prev = tf.image.resize(prev, (self.sizes[i + 1][0], self.sizes[i + 1][1]))
		
		return prev

	def test(self):
		result = self.G(tf.random.uniform([self.args.batch_size, self.args.z_dim], -1., 1.), training=False)
		imsave(os.path.join(self.args.result_dir, 'result.jpg'), montage(imdenorm(result.numpy())))

	def load_model(self, all_module=False):
		self.G = tk.models.load_model(os.path.join(self.args.save_dir, 'G.h5'))
		
		if all_module:
			self.D = tk.models.load_model(os.path.join(self.args.save_dir, 'D.h5'))

	def save_model(self, scale):
		self.G.save(os.path.join(self.args.save_dir, f'G_{scale}.h5'))
		self.D.save(os.path.join(self.args.save_dir, f'D_{scale}.h5'))

		with open(os.path.join(self.args.save_dir, 'z_fixed.pkl'), 'wb') as fw:
			pickle.dump([self.z_fixed.numpy(), self.noise_weights], fw)