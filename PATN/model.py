# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as tk
from tensorflow.keras.applications import vgg19, VGG19
import numpy as np
import time
from dataloader import Dataloader
import sys
sys.path.append('..')
from ops import *
from utils import *

class decoder(tk.Model):
	def __init__(self, img_nc):
		super(decoder, self).__init__()
		self.img_up = [tk.Sequential([Resblock(512), Resblock(512), Resblock(512), Relu(), Deconv2d(128, 3), IN(), Relu()]),
					   tk.Sequential([Resblock(256), Resblock(256), Resblock(256), Relu(), Deconv2d(64,  3), IN(), Relu()]),
					   tk.Sequential([Resblock(128), Resblock(128), Resblock(128), Relu(), Conv2d(64, 3, 1), IN(), Relu(), 
					   Conv2d(img_nc, 7, 1, activation='tanh')])]

	def call(self, h_img, features):
		for i, S in enumerate(self.img_up):
			h_img = S(tf.concat([h_img, features[-1 - i]], -1))
		return h_img

class Model(tk.Model):
	def __init__(self, args):
		super(Model, self).__init__()
		self.args = args

	def attn_conv(self, concat, last_norm):
		layers = [Conv2d(512 if concat else 256, 3, 1), IN(), Relu(), Dropout(0.5), Conv2d(256, 3, 1)]

		if last_norm:
			layers += [IN()]
		
		return tk.Sequential(layers)

	def attn_block(self, img, pose, concat):
		h_img = self.attn_conv(False, True)(img)
		h_pose = self.attn_conv(concat, False)(pose)

		res = h_img * sigmoid(h_pose)
		h_img = img + res
		h_pose = tf.concat([h_img, h_pose], -1)
		
		return h_img, h_pose

	def encoder(self):
		img = Input((self.args.img_h, self.args.img_w, self.args.img_nc))
		pose = Input((self.args.img_h, self.args.img_w, 2 * self.args.n_kps))
		
		img_down = [tk.Sequential([Conv2d(64, 7, 1), IN(), Relu()]),
					tk.Sequential([Conv2d(128, 3, 2), IN(), Relu()]),
					tk.Sequential([Conv2d(256, 3, 2), IN(), Relu()])]

		features = []
		h_img = img
		for S in img_down:
			h_img = S(h_img)
			features.append(h_img)

		h_pose = tk.Sequential([pose,
			Conv2d(64,  7, 1), IN(), Relu(),
			Conv2d(128, 3, 2), IN(), Relu(),
			Conv2d(256, 3, 2), IN(), Relu()]).output

		for i in range(self.args.n_attn_block):
			h_img, h_pose = self.attn_block(h_img, h_pose, False if i == 0 else True)

		return tk.Model([img, pose], [h_img, features])

	def discriminator(self, input_nc):
		return tk.Sequential([
			Input((self.args.img_h, self.args.img_w, input_nc)),
			Conv2d(64,  7, 1), IN(), Relu(),
			Conv2d(128, 3, 2), IN(), Relu(),
			Conv2d(256, 3, 2), IN(), Relu(),
			Resblock(256, dropout=True), Resblock(256, dropout=True), Resblock(256, dropout=True)])

	def build_model(self):
		if self.args.phase == 'train':
			self.iter = iter(Dataloader(self.args).loader)
			# self.features = None

			self.E = self.encoder()
			self.G = decoder(self.args.img_nc)
			self.DI = self.discriminator(2 * self.args.img_nc)
			self.DK = self.discriminator(self.args.img_nc + self.args.n_kps)

			boundaries = []
			values = [self.args.lr]
			for i in range(self.args.decay_epochs):
				boundaries.append(self.args.iteration * (self.args.epochs + i))
				values.append(self.args.lr - i * self.args.lr / self.args.decay_epochs)
			
			lr = tk.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
			self.optimizer_g = tk.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)
			self.optimizer_di = tk.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)
			self.optimizer_dk = tk.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)

			self.vgg = VGG19(include_top=False, weights='imagenet')
			self.vgg.trainable = False
			self.vgg = tk.Model(self.vgg.input, self.vgg.get_layer(self.args.vgg_layer).output)
			self.roi_output_sizes = [[16, 16], [16, 7], [16, 7], [32, 32], [32, 7], [32, 7], [16, 16]]

			self.summary_writer = tf.summary.create_file_writer(self.args.log_dir)
		
		elif self.args.phase == 'test':
			self.load_model()

	@tf.function
	def segments_seperate_style_loss(self, fake, real, bbox):
		loss_rec = self.args.w_rec * l1_loss(fake, real)

		indices = [[] for i in range(self.args.n_body_part)]
		boxes = [[] for i in range(self.args.n_body_part)]
		for i in range(self.args.batch_size):
			for j in range(self.args.n_body_part):
				box = bbox[i, j, :]

				if box[0] == 0 and box[1] == 0 and box[2] == 0 and box[3] == 0:
					continue

				indices[j].append(i)
				boxes[j].append(box)

		real_feature = self.vgg(vgg19.preprocess_input((real + 1.) * 127.5))
		fake_feature = self.vgg(vgg19.preprocess_input((fake + 1.) * 127.5))
		loss_per = self.args.w_per * l1_loss(fake_feature, real_feature)

		img_size = (self.args.img_h, self.args.img_w)
		loss_style = 0
		for j in range(self.args.n_body_part):
			if len(boxes[j]) == 0:
				continue

			fake_feature_part = roi_align(fake_feature, boxes[j], indices[j], img_size, self.roi_output_sizes[j])
			real_feature_part = roi_align(real_feature, boxes[j], indices[j], img_size, self.roi_output_sizes[j])
			loss_style += self.args.w_style * gram_mse(fake_feature_part, real_feature_part)

		return loss_rec, loss_per, loss_style

	def blend(self, k, feature, a_bbox, b_bbox):
		self.features[k].assign(tf.zeros(feature.shape))
		a_bbox, b_bbox = tf.cast(a_bbox + 0.5, 'int32'), tf.cast(b_bbox + 0.5, 'int32')

		for i in range(self.args.batch_size):
			for j in range(self.args.n_body_part):
				ax0, ay0, ax1, ay1 = a_bbox[i, j, 0], a_bbox[i, j, 1], a_bbox[i, j, 2], a_bbox[i, j, 3]
				bx0, by0, bx1, by1 = b_bbox[i, j, 0], b_bbox[i, j, 1], b_bbox[i, j, 2], b_bbox[i, j, 3]

			if ax0 == 0 and ay0 == 0 and ax1 == 0 and ay1 == 0:
				continue

			if bx0 == 0 and by0 == 0 and bx1 == 0 and by1 == 0:
				continue
			
			feat = feature[i, ay0: ay1, ax0: ax1, :]
			bH, bW = by1 - by0, bx1 - bx0
			
			if bW > 1:
				x_map = tf.tile(tf.reshape(-1 + tf.cast(2 / (bW - 1), 'float32') * tf.cast(tf.range(bW), 'float32'), [1, 1, bW]), [1, bH, 1])
			else:
				x_map = tf.ones([1, bH, 1])
			
			if bH > 1:
				y_map = tf.tile(tf.reshape(-1 + tf.cast(2 / (bH - 1), 'float32') * tf.cast(tf.range(bH), 'float32'), [1, bH, 1]), [1, 1, bW])
			else:
				y_map = tf.ones([1, 1, bW])

			feat = grid_sample(tf.expand_dims(feat, 0), x_map, y_map)[0]
			self.features[k][i, by0: by1, bx0: bx1, :].assign(tf.maximum(feat, self.features[k][i, by0: by1, bx0: bx1, :]))

		return self.features[k]

	@tf.function
	def train_one_step(self, A_img, B_img, A_kps, B_kps, A_bbox, B_bbox):
		with tf.GradientTape() as tape_g, tf.GradientTape() as tape_di, tf.GradientTape() as tape_dk:
			h_img, features = self.E([A_img, tf.concat([A_kps, B_kps], -1)], training=True)
			# if self.features == None:
			# 	self.features = [tf.Variable(tf.zeros(features[i].shape), trainable=False) for i in range(len(features))]

			# features = [self.blend(i, features[i], A_bbox / 2 ** i, B_bbox / 2 ** i) for i in range(len(features))]
			fake = self.G(h_img, features, training=True)
			
			d_fake_i = self.DI(tf.concat([fake, A_img], -1), training=True)
			loss_g_adv_i = generator_loss(d_fake_i, self.args.gan_type)

			d_fake_k = self.DK(tf.concat([fake, B_kps], -1), training=True)
			loss_g_adv_k = generator_loss(d_fake_k, self.args.gan_type)
			
			loss_g_adv = self.args.w_adv * (loss_g_adv_i + loss_g_adv_k) / 2

			loss_g_rec, loss_g_per, loss_g_style = self.segments_seperate_style_loss(fake, B_img, B_bbox)
			loss_g = loss_g_adv + loss_g_rec + loss_g_per + loss_g_style

			d_real_i = self.DI(tf.concat([B_img, A_img], -1), training=True)
			loss_d_i = self.args.w_adv * discriminator_loss(d_real_i, d_fake_i, self.args.gan_type)

			d_real_k = self.DK(tf.concat([B_img, B_kps], -1), training=True)
			loss_d_k = self.args.w_adv * discriminator_loss(d_real_k, d_fake_k, self.args.gan_type)

		self.vars_g = self.E.trainable_variables + self.G.trainable_variables
		self.vars_di = self.DI.trainable_variables
		self.vars_dk = self.DK.trainable_variables
		self.optimizer_g.apply_gradients(zip(tape_g.gradient(loss_g, self.vars_g), self.vars_g))
		self.optimizer_di.apply_gradients(zip(tape_di.gradient(loss_d_i, self.vars_di), self.vars_di))
		self.optimizer_dk.apply_gradients(zip(tape_dk.gradient(loss_d_k, self.vars_dk), self.vars_dk))

		return {'loss_g_adv_i': loss_g_adv_i, 'loss_g_adv_k': loss_g_adv_k, 'loss_g_rec': loss_g_rec, 'loss_g_per': loss_g_per,
				'loss_g_style': loss_g_style, 'loss_d_i': loss_d_i, 'loss_d_k': loss_d_k}

	def train(self):
		start_time = time.time()
		step = 0
		img_h, img_w = self.args.img_h, self.args.img_w
		for e in range(self.args.epochs + self.args.decay_epochs):
			for i in range(self.args.iteration):
				A_img, B_img, A_kps, B_kps, A_bbox, B_bbox = next(self.iter)

				item = self.train_one_step(A_img, B_img, A_kps, B_kps, A_bbox, B_bbox)
				print('epoch: [%3d/%3d] iter: [%6d/%6d] time: %.2f' % (e, self.args.epochs + self.args.decay_epochs, 
					i, self.args.iteration, time.time() - start_time))
				step += 1

				if step % self.args.log_freq == 0:
					with self.summary_writer.as_default():
						tf.summary.scalar('loss_g_adv_i', item['loss_g_adv_i'], step=step)
						tf.summary.scalar('loss_g_adv_k', item['loss_g_adv_k'], step=step)
						tf.summary.scalar('loss_g_rec',   item['loss_g_rec'],   step=step)
						tf.summary.scalar('loss_g_per',   item['loss_g_per'],   step=step)
						tf.summary.scalar('loss_g_style', item['loss_g_style'], step=step)
						tf.summary.scalar('loss_d_i',     item['loss_d_i'],     step=step)
						tf.summary.scalar('loss_d_k',     item['loss_d_k'],     step=step)

			h_img, features = self.E([A_img, tf.concat([A_kps, B_kps], -1)], training=False)
			# features = [self.blend(features[i], A_bbox / 2 ** i, B_bbox / 2 ** i) for i in range(len(features))]
			fake = self.G(h_img, features, training=False)

			fake = imdenorm(fake.numpy())
			A_img = imdenorm(A_img.numpy())
			B_img = imdenorm(B_img.numpy())

			sample = np.zeros((self.args.batch_size * img_h, 5 * img_w, self.args.img_nc))
			for i in range(self.args.batch_size):
				sample[i * img_h: (i + 1) * img_h, 0 * img_w: 1 * img_w] = A_img[i]
				sample[i * img_h: (i + 1) * img_h, 1 * img_w: 2 * img_w] = self.htmap_to_img(A_kps[i])
				sample[i * img_h: (i + 1) * img_h, 2 * img_w: 3 * img_w] = B_img[i]
				sample[i * img_h: (i + 1) * img_h, 3 * img_w: 4 * img_w] = self.htmap_to_img(B_kps[i])
				sample[i * img_h: (i + 1) * img_h, 4 * img_w: 5 * img_w] = fake[i]

			imsave(os.path.join(self.args.sample_dir, f'{e}.jpg'), sample)
			self.save_model()

	def test(self):
		pass

	def htmap_to_img(self, kps):
		return np.clip(np.expand_dims(np.sum(kps, -1), -1), 0, 1)

	def load_model(self, all_module=False):
		self.E = self.encoder()
		self.E.load_weights(os.path.join(self.args.save_dir, 'E.h5'))
		self.G = decoder(self.args.img_nc)
		self.G.load_weights(os.path.join(self.args.save_dir, 'G.h5'))
			
		if all_module:
			self.DI = self.discriminator(2 * self.args.img_nc)
			self.DI.load_weights(os.path.join(self.args.save_dir, 'DI.h5'))
			self.DK = self.discriminator(self.args.img_nc + self.args.n_kps)
			self.DK.load_weights(os.path.join(self.args.save_dir, 'DK.h5'))

	def save_model(self):
		self.E.save_weights(os.path.join(self.args.save_dir, 'E.h5'))
		self.G.save_weights(os.path.join(self.args.save_dir, 'G.h5'))
		self.DI.save_weights(os.path.join(self.args.save_dir, 'DI.h5'))
		self.DK.save_weights(os.path.join(self.args.save_dir, 'DK.h5'))