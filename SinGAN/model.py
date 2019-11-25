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
			self.filters = [min(self.args.num_filter * np.power(2, i // 4), 128) for i in range(self.num_scale)]
			self.Gs = [self.generator(self.sizes[i][0], self.sizes[i][1], self.sizes[i][2], self.filters[i]) for i in range(self.num_scale)]
			self.Ds = [self.discriminator(self.sizes[i][0], self.sizes[i][1], self.sizes[i][2], self.filters[i]) for i in range(self.num_scale)]
			self.z_fixed = tf.random.normal(self.imgs[0].shape)
			self.noise_weights = []
		
		elif self.args.phase == 'test':
			self.load_model()

	def train(self):
		start_time = time.time()
		for scale in range(self.num_scale):
			img = self.imgs[scale]
			imsave(os.path.join(self.args.save_dir, f'real_{scale}.jpg'), imdenorm(img.numpy()[0]))

			if scale > 0 and self.filters[scale] == self.filters[scale - 1]:
				self.Gs[scale].load_weights(os.path.join(self.args.save_dir, f'G_{scale - 1}.h5'))
				self.Ds[scale].load_weights(os.path.join(self.args.save_dir, f'D_{scale - 1}.h5'))

			lr = tk.optimizers.schedules.ExponentialDecay(self.args.lr, decay_steps=self.args.decay_steps, decay_rate=self.args.decay_rate)
			self.optimizer_g = tk.optimizers.Adam(learning_rate=lr, beta_1=0.5)
			self.optimizer_d = tk.optimizers.Adam(learning_rate=lr, beta_1=0.5)

			rec_prev = 0.
			for s in range(scale):
				zr = self.z_fixed if s == 0 else rec_prev
				rec_prev = self.Gs[s]([zr, rec_prev], training=False)
				rec_prev = tf.image.resize(rec_prev, (self.sizes[s + 1][0], self.sizes[s + 1][1]))

			noise_weight = 1. if scale == 0 else self.args.noise_weight * tf.sqrt(l2_loss(img, rec_prev))
			self.noise_weights.append(noise_weight)

			for i in range(self.args.iteration):
				z = tf.random.normal(self.imgs[scale].shape)

				fake_prev = 0.
				for s in range(scale):
					fake_prev = self.Gs[s]([self.noise_weights[s] * tf.random.normal(self.imgs[s].shape) + fake_prev, fake_prev], training=False)
					fake_prev = tf.image.resize(fake_prev, (self.sizes[s + 1][0], self.sizes[s + 1][1]))

				for j in range(self.args.D_step):
					with tf.GradientTape() as tape_d:
						d_real = self.Ds[scale](img, training=True)
						fake = self.Gs[scale]([noise_weight * z + fake_prev, fake_prev], training=True)
						d_fake = self.Ds[scale](fake, training=True)
						loss_d_adv = discriminator_loss(d_real, d_fake, self.args.gan_type)

						alpha = tf.random.uniform([1, 1, 1, 1], 0., 1.)
						inter_sample = alpha * img + (1 - alpha) * fake
						loss_d_gp = gradient_penalty(self.Ds[scale], inter_sample, self.args.w_gp)
						
						loss_d = loss_d_adv + loss_d_gp

					self.optimizer_d.apply_gradients(zip(tape_d.gradient(loss_d, self.Ds[scale].trainable_variables), self.Ds[scale].trainable_variables))

				for j in range(self.args.G_step):
					with tf.GradientTape() as tape_g:
						loss_g_adv = generator_loss(d_fake, self.args.gan_type)

						zr = self.z_fixed if scale == 0 else rec_prev
						rec = self.Gs[scale]([zr, rec_prev], training=True)
						loss_g_rec = self.args.w_rec * l2_loss(rec, img)

						loss_g = loss_g_adv + loss_g_rec

					self.optimizer_g.apply_gradients(zip(tape_g.gradient(loss_g, self.Gs[scale].trainable_variables), self.Gs[scale].trainable_variables))

				print('scale: [%d/%d] iter: [%4d/%4d] time: %.2f loss_g_adv: %.4f, loss_g_rec: %.4f, loss_d_adv: %.4f, loss_d_gp: %.4f' % (
					scale, self.num_scale, i, self.args.iteration, time.time() - start_time, loss_g_adv.numpy(), loss_g_rec.numpy(), loss_d_adv.numpy(), loss_d_gp.numpy()))

				if (i + 1) % self.args.sample_freq == 0:
					imsave(os.path.join(self.args.save_dir, f'fake_{scale}_{i + 1}.jpg'), imdenorm(fake.numpy()[0]))
					imsave(os.path.join(self.args.save_dir, f'rec_{scale}_{i + 1}.jpg'), imdenorm(rec.numpy()[0]))

			self.save_model(scale)

	def test(self):
		result = self.G(tf.random.uniform([self.args.batch_size, self.args.z_dim], -1., 1.), training=False)
		imsave(os.path.join(self.args.result_dir, 'result.jpg'), montage(imdenorm(result.numpy())))

	def load_model(self, all_module=False):
		self.G = tk.models.load_model(os.path.join(self.args.save_dir, 'G.h5'))
		
		if all_module:
			self.D = tk.models.load_model(os.path.join(self.args.save_dir, 'D.h5'))

	def save_model(self, scale):
		self.Gs[scale].save(os.path.join(self.args.save_dir, f'G_{scale}.h5'))
		self.Ds[scale].save(os.path.join(self.args.save_dir, f'D_{scale}.h5'))

		with open(os.path.join(self.args.save_dir, 'z_fixed_and_noise_weights.pkl'), 'wb') as fw:
			pickle.dump([self.z_fixed.numpy(), self.noise_weights], fw)