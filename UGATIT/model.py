# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as tk
import numpy as np
import time, pickle
from dataloader import Dataloader
import sys
sys.path.append('..')
from ops import *
from utils import *

class Model(object):
	def __init__(self, args):
		self.args = args

	def cam(self, h, filters, sn):
		dense_with_w = Dense_with_w(sn=sn)
		cam_gap_logit, cam_gap_weight = dense_with_w(global_avg_pooling(h))
		h_gap = h * cam_gap_weight
		cam_gmp_logit, cam_gmp_weight = dense_with_w(global_max_pooling(h))
		h_gmp = h * cam_gmp_weight

		cam_logit = tf.concat([cam_gap_logit, cam_gmp_logit], -1)
		h = relu(Conv2d(filters, 1, 1)(tf.concat([h_gap, h_gmp], -1)))
		heatmap = tf.reduce_sum(h, axis=-1)

		return cam_logit, h, heatmap

	def generator(self):
		x = Input((self.args.img_size, self.args.img_size, self.args.img_nc))

		h = tk.Sequential([x,
			Conv2d(64,  7, 1), IN(), Relu(),
			Conv2d(128, 3, 2), IN(), Relu(),
			Conv2d(256, 3, 2), IN(), Relu(),
			Resblock(256), Resblock(256), Resblock(256), Resblock(256)]).output

		cam_logit, h, heatmap = self.cam(h, filters=256, sn=False)

		mlp = tk.Sequential([Flatten(), Dense(256), Relu(), Dense(256), Relu()])(h) # more params
		# mlp = tk.Sequential([Dense(256), Relu(), Dense(256), Relu()])(global_avg_pooling(h)) # fewer params
		gamma = tf.reshape(Dense(256)(mlp), [-1, 1, 1, 256])
		beta  = tf.reshape(Dense(256)(mlp), [-1, 1, 1, 256])

		for i in range(4):
			h = AdaLINResblock(256)(h, gamma, beta)

		h = tk.Sequential([
			UpSample(), Conv2d(128, 3, 1), LIN(), Relu(),
			UpSample(), Conv2d(64,  3, 1), LIN(), Relu(),
			Conv2d(self.args.img_nc, 7, 1, activation='tanh')])(h)

		return tk.Model(x, [h, cam_logit, heatmap])

	def sub_discriminator(self, n_dis):
		x = Input((self.args.img_size, self.args.img_size, self.args.img_nc))

		filters = 64
		h = lrelu(Conv2d(filters, 4, 2, sn=True)(x))

		for i in range(1, n_dis - 1):
			filters = min(filters * 2, 512)
			h = lrelu(Conv2d(filters, 4, 2, sn=True)(h))

		filters = min(filters * 2, 512)
		h = lrelu(Conv2d(filters, 4, 1, sn=True)(h))

		cam_logit, h, heatmap = self.cam(h, filters=256, sn=True)
		h = Conv2d(1, 4, 1, sn=True)(h)

		return tk.Model(x, [h, cam_logit, heatmap])

	def discriminator(self):
		x = Input((self.args.img_size, self.args.img_size, self.args.img_nc))
		
		local_h, local_cam_logit, local_heatmap = self.sub_discriminator(4)(x)
		global_h, global_cam_logit, global_heatmap = self.sub_discriminator(6)(x)
		
		return tk.Model(x, [[local_h, global_h], [local_cam_logit, global_cam_logit], local_heatmap, global_heatmap])

	def build_model(self):
		dataloader = Dataloader(self.args)
		if self.args.phase == 'train':
			self.iter_A, self.iter_B = iter(dataloader.loader_A), iter(dataloader.loader_B)

			self.Ga, self.Gb = self.generator(), self.generator()
			self.Da, self.Db = self.discriminator(), self.discriminator()

			boundaries = []
			values = [self.args.lr]
			for i in range(self.args.decay_epochs):
				boundaries.append(self.args.iteration * (self.args.epochs + i))
				values.append(self.args.lr - i * self.args.lr / self.args.decay_epochs)

			lr = tk.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
			self.optimizer_g = tk.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)
			self.optimizer_d = tk.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)

			self.summary_writer = tf.summary.create_file_writer(self.args.log_dir)
		
		elif self.args.phase == 'test':
			self.A_loader, self.B_loader = dataloader.A_loader, dataloader.B_loader
			self.N_A = self.A_loader.reduce(0, lambda x, _: x + 1)
			self.N_B = self.B_loader.reduce(0, lambda x, _: x + 1)
			self.load()

	def cam_loss(self, source, non_source):
		bce = tk.losses.BinaryCrossentropy(from_logits=True)
		
		identity_loss = bce(tf.ones_like(source), source)
		non_identity_loss = bce(tf.zeros_like(non_source), non_source)
		
		return identity_loss + non_identity_loss

	@tf.function
	def train_step(self, A, B):
		with tf.GradientTape() as tape_g, tf.GradientTape() as tape_d:
			fake_b, cam_b, _ = self.Gb(A)
			fake_a, cam_a, _ = self.Ga(B)

			cyc_a, _, _ = self.Ga(fake_b)
			cyc_b, _, _ = self.Gb(fake_a)

			rec_a, cam_aa, _ = self.Ga(A)
			rec_b, cam_bb, _ = self.Gb(B)

			d_real_a, d_real_a_cam, _, _ = self.Da(A)
			d_real_b, d_real_b_cam, _, _ = self.Db(B)
			d_fake_a, d_fake_a_cam, _, _ = self.Da(fake_a)
			d_fake_b, d_fake_b_cam, _, _ = self.Db(fake_b)

			loss_g_adv = self.args.w_adv * tf.reduce_sum([
				generator_loss(d_fake_a, self.args.gan_type, True), generator_loss(d_fake_a_cam, self.args.gan_type, True),
				generator_loss(d_fake_b, self.args.gan_type, True), generator_loss(d_fake_b_cam, self.args.gan_type, True)])
			loss_d_adv = self.args.w_adv * tf.reduce_sum([
				discriminator_loss(d_real_a, d_fake_a, self.args.gan_type, True),
				discriminator_loss(d_real_a_cam, d_fake_a_cam, self.args.gan_type, True),
				discriminator_loss(d_real_b, d_fake_b, self.args.gan_type, True),
				discriminator_loss(d_real_b_cam, d_fake_b_cam, self.args.gan_type, True)])

			loss_g_cyc = self.args.w_cyc * (l1_loss(A, cyc_a) + l1_loss(B, cyc_b))
			loss_g_rec = self.args.w_rec * (l1_loss(A, rec_a) + l1_loss(B, rec_b))
			loss_g_cam = self.args.w_cam * (self.cam_loss(cam_a, cam_aa) + self.cam_loss(cam_b, cam_bb))

			loss_g = loss_g_adv + loss_g_cyc + loss_g_rec + loss_g_cam
			loss_d = loss_d_adv

		vars_g = self.Ga.trainable_variables + self.Gb.trainable_variables
		vars_d = self.Da.trainable_variables + self.Db.trainable_variables
		self.optimizer_g.apply_gradients(zip(tape_g.gradient(loss_g, vars_g), vars_g))
		self.optimizer_d.apply_gradients(zip(tape_d.gradient(loss_d, vars_d), vars_d))

		return {'loss_g_adv': loss_g_adv, 'loss_g_cyc': loss_g_cyc, 'loss_g_rec': loss_g_rec, 'loss_g_cam': loss_g_cam,
				'loss_d_adv': loss_d_adv, 'fake_a': fake_a, 'fake_b': fake_b}

	def train(self):
		start_time = time.time()
		step = 0
		img_size = self.args.img_size
		for e in range(self.args.epochs + self.args.decay_epochs):
			for i in range(self.args.iteration):
				A, B = next(self.iter_A), next(self.iter_B)

				item = self.train_step(A, B)
				print('epoch: [%3d/%3d] iter: [%6d/%6d] time: %.2f' % (e, self.args.epochs + self.args.decay_epochs, 
					i, self.args.iteration, time.time() - start_time))
				step += 1
				
				if step % self.args.log_freq == 0:
					with self.summary_writer.as_default():
						tf.summary.scalar('loss_g_adv', item['loss_g_adv'], step=step)
						tf.summary.scalar('loss_g_cyc', item['loss_g_cyc'], step=step)
						tf.summary.scalar('loss_g_rec', item['loss_g_rec'], step=step)
						tf.summary.scalar('loss_g_cam', item['loss_g_cam'], step=step)
						tf.summary.scalar('loss_d_adv', item['loss_d_adv'], step=step)

			A, B = imdenorm(A.numpy()), imdenorm(B.numpy())
			fake_a, fake_b = imdenorm(item['fake_a'].numpy()), imdenorm(item['fake_b'].numpy())
			sample = np.ones((self.args.batch_size * img_size, 4 * img_size, self.args.img_nc))

			for i in range(self.args.batch_size):
				sample[i * img_size: (i + 1) * img_size, 0 * img_size: 1 * img_size] = A[i]
				sample[i * img_size: (i + 1) * img_size, 1 * img_size: 2 * img_size] = fake_b[i]
				sample[i * img_size: (i + 1) * img_size, 2 * img_size: 3 * img_size] = B[i]
				sample[i * img_size: (i + 1) * img_size, 3 * img_size: 4 * img_size] = fake_a[i]

			imsave(os.path.join(self.args.sample_dir, f'{e}.jpg'), sample)
			self.save()

	def test(self):
		img_size = self.args.img_size

		A2B = np.ones((2 * img_size, self.N_A * img_size, self.args.img_nc))
		for i, A in enumerate(self.A_loader):
			fake_B, _, _ = self.Gb(tf.expand_dims(A, 0))
			A2B[:img_size, i * img_size: (i + 1) * img_size] = imdenorm(A.numpy())
			A2B[img_size:, i * img_size: (i + 1) * img_size] = imdenorm(fake_B[0].numpy())

		imsave(os.path.join(self.args.result_dir, 'A2B.jpg'), A2B)

		B2A = np.ones((2 * img_size, self.N_B * img_size, self.args.img_nc))
		for i, B in enumerate(self.B_loader):
			fake_A, _, _ = self.Ga(tf.expand_dims(B, 0))
			B2A[:img_size, i * img_size: (i + 1) * img_size] = imdenorm(B.numpy())
			B2A[img_size:, i * img_size: (i + 1) * img_size] = imdenorm(fake_A[0].numpy())
			
		imsave(os.path.join(self.args.result_dir, 'B2A.jpg'), B2A)

	def load(self, all_module=False):
		self.Ga = self.generator()
		self.Ga.load_weights(os.path.join(self.args.save_dir, 'Ga.h5'))
		self.Gb = self.generator()
		self.Gb.load_weights(os.path.join(self.args.save_dir, 'Gb.h5'))
		
		if all_module:
			self.Da = self.discriminator()
			self.Da.load_weights(os.path.join(self.args.save_dir, 'Da.h5'))
			self.Db = self.discriminator()
			self.Db.load_weights(os.path.join(self.args.save_dir, 'Db.h5'))

	def save(self):
		self.Ga.save_weights(os.path.join(self.args.save_dir, 'Ga.h5'))
		self.Gb.save_weights(os.path.join(self.args.save_dir, 'Gb.h5'))
		self.Da.save_weights(os.path.join(self.args.save_dir, 'Da.h5'))
		self.Db.save_weights(os.path.join(self.args.save_dir, 'Db.h5'))