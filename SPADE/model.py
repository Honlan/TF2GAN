# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as tk
from tensorflow.keras.applications import vgg19, VGG19
import numpy as np
import time, pickle
from dataloader import Dataloader
import sys
sys.path.append('..')
from ops import *
from utils import *

class decoder(tk.Model):
	def __init__(self):
		super(decoder, self).__init__()
		self.dense = Dense(4 * 4 * 1024)
		self.conv = Conv2d(3, 3, 1, activation='tanh')
		self.spade_resblocks = []
		filters = [1024, 1024, 1024, 512, 256, 128, 64]
		for i in range(1, len(filters)):
			self.spade_resblocks.append(SpadeResblock(filters[i - 1], filters[i], sn=True))

	def call(self, m, mean, var, random_style=False):
		h = tf.random.normal([m.shape[0], 256]) if random_style else z_sample(mean, var)
		h = tf.reshape(self.dense(h), [-1, 4, 4, 1024])

		for i in range(len(self.spade_resblocks)):
			h = up_sample(self.spade_resblocks[i](m, h))

		return self.conv(lrelu(h))

class Model(object):
	def __init__(self, args):
		self.args = args

	def encoder(self):
		x = Input((self.args.img_size, self.args.img_size, self.args.img_nc))

		h = tk.Sequential([x,
			Conv2d(64,  3, 2, sn=True), IN(), Lrelu(),
			Conv2d(128, 3, 2, sn=True), IN(), Lrelu(),
			Conv2d(256, 3, 2, sn=True), IN(), Lrelu(),
			Conv2d(512, 3, 2, sn=True), IN(), Lrelu(),
			Conv2d(512, 3, 2, sn=True), IN(), Lrelu(),
			Conv2d(512, 3, 2, sn=True), IN(), Lrelu()]).output

		mean = Dense(256, sn=True)(flatten(h))
		var  = Dense(256, sn=True)(flatten(h))

		return tk.Model(x, [mean, var])

	def discriminator(self, input_nc):
		x_init = Input((self.args.img_size, self.args.img_size, input_nc))
		x = x_init
		
		logits = []
		for i in range(self.args.dis_scale):
			sub_logits = []
			filters = 64
			h = lrelu(Conv2d(filters, 4, 2)(x))
			sub_logits.append(h)

			for j in range(1, self.args.dis_layer):
				strides = 1 if j == self.args.dis_layer - 1 else 2
				h = lrelu(IN()(Conv2d(filters * 2, 4, strides, sn=True)(h)))
				sub_logits.append(h)
				filters = min(filters * 2, 512)

			h = Conv2d(1, 4, 1, sn=True)(h)
			sub_logits.append(h)
			logits.append(sub_logits)
			x = AvgPool2d()(x)

		return tk.Model(x_init, logits)

	def build_model(self):
		if self.args.phase == 'train':
			self.iter = iter(Dataloader(self.args).loader)

			self.E = self.encoder()
			self.G = decoder()
			self.D = self.discriminator(self.args.img_nc + self.args.label_nc)

			boundaries = []
			values_g = [self.args.lr / 2]
			values_d = [self.args.lr * 2]
			for i in range(self.args.decay_epochs):
				boundaries.append(self.args.iteration * (self.args.epochs + i))
				values_g.append((self.args.lr - i * self.args.lr / self.args.decay_epochs) / 2)
				values_d.append((self.args.lr - i * self.args.lr / self.args.decay_epochs) * 2)

			lr_g = tk.optimizers.schedules.PiecewiseConstantDecay(boundaries, values_g)
			lr_d = tk.optimizers.schedules.PiecewiseConstantDecay(boundaries, values_d)
			self.optimizer_g = tk.optimizers.Adam(learning_rate=lr_g, beta_1=0.0, beta_2=0.9)
			self.optimizer_d = tk.optimizers.Adam(learning_rate=lr_d, beta_1=0.0, beta_2=0.9)

			self.vgg = VGG19(include_top=False, weights='imagenet')
			self.vgg.trainable = False
			self.vgg_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
			self.vgg_weights = [1. / 32, 1. / 16, 1. / 8, 1. / 4, 1.]
			self.vgg = tk.Model(self.vgg.input, [self.vgg.get_layer(layer).output for layer in self.vgg_layers])

			self.summary_writer = tf.summary.create_file_writer(self.args.log_dir)
		
		elif self.args.phase == 'test':
			self.load()

	@tf.function
	def vgg_loss(self, real, fake):
		real_features = self.vgg(vgg19.preprocess_input((real + 1.) * 127.5))
		fake_features = self.vgg(vgg19.preprocess_input((fake + 1.) * 127.5))
		return tf.reduce_sum([self.vgg_weights[i] * l1_loss(
			real_features[i] / 127.5, fake_features[i] / 127.5) for i in range(len(real_features))])

	@tf.function
	def train_step(self, img, label):
		with tf.GradientTape() as tape_g, tf.GradientTape() as tape_d:
			mean, var = self.E(img)
			fake = self.G(label, mean, var, random_style=False)
			
			d_real = self.D(tf.concat([img, label], -1))
			d_fake = self.D(tf.concat([fake, label], -1))
			loss_g_adv = self.args.w_adv * generator_loss([d[-1] for d in d_fake], self.args.gan_type, True)
			loss_d_adv = self.args.w_adv * discriminator_loss([d[-1] for d in  d_real], [d[-1] for d in d_fake], self.args.gan_type, True)
			
			loss_g_vgg = self.args.w_vgg * self.vgg_loss(img, fake)
			loss_g_fm = self.args.w_fm * feature_loss(d_real, d_fake)
			loss_g_kl = self.args.w_kl * kl_loss(mean, var)

			loss_g = loss_g_adv + loss_g_vgg + loss_g_fm + loss_g_kl
			loss_d = loss_d_adv

		vars_g = self.E.trainable_variables + self.G.trainable_variables
		vars_d = self.D.trainable_variables
		self.optimizer_g.apply_gradients(zip(tape_g.gradient(loss_g, vars_g), vars_g))
		self.optimizer_d.apply_gradients(zip(tape_d.gradient(loss_d, vars_d), vars_d))

		return {'loss_g_adv': loss_g_adv, 'loss_g_vgg': loss_g_vgg, 'loss_g_fm': loss_g_fm, 'loss_g_kl': loss_g_kl,
				'loss_d_adv': loss_d_adv, 'mean': mean, 'var': var, 'fake': fake}

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
						tf.summary.scalar('loss_g_adv', item['loss_g_adv'], step=step)
						tf.summary.scalar('loss_g_vgg', item['loss_g_vgg'], step=step)
						tf.summary.scalar('loss_g_fm',  item['loss_g_fm'],  step=step)
						tf.summary.scalar('loss_g_kl',  item['loss_g_kl'],  step=step)
						tf.summary.scalar('loss_d_adv', item['loss_d_adv'], step=step)

			mean, var, fake = item['mean'], item['var'], item['fake']
			img, fake = imdenorm(img.numpy()), imdenorm(fake.numpy())
			sample = np.zeros((self.args.batch_size * img_size, (3 + self.args.n_random) * img_size, self.args.img_nc))

			for i in range(self.args.batch_size):
				sample[i * img_size: (i + 1) * img_size, 0 * img_size: 1 * img_size] = img[i]
				sample[i * img_size: (i + 1) * img_size, 1 * img_size: 2 * img_size] = self.multi_to_one(label[i].numpy())
				sample[i * img_size: (i + 1) * img_size, 2 * img_size: 3 * img_size] = fake[i]
			
			for j in range(self.args.n_random):
				random = self.G(label, None, None, random_style=True)
				random = imdenorm(random.numpy())
				for i in range(self.args.batch_size):
					sample[i * img_size: (i + 1) * img_size, (3 + j) * img_size: (4 + j) * img_size] = random[i]

			imsave(os.path.join(self.args.sample_dir, f'{e}.jpg'), sample)
			self.save(label[:1].numpy())

	def test(self):
		img_size = self.args.img_size
		if self.args.test_mode == 'random':
			label_paths = glob(os.path.join('dataset', self.args.dataset_name, self.args.test_label_dir, '*'))
			label = np.array([load_label(label_path) for label_path in label_paths])

			result = np.zeros((len(label_paths) * img_size, (self.args.n_random + 1) * img_size, self.args.img_nc))
			for j in range(self.args.n_random):
				random = self.G(label, None, None, random_style=True)
				random = imdenorm(random.numpy())
				
				for i in range(len(label_paths)):
					result[i * img_size: (i + 1) * img_size, (j + 1) * img_size: (j + 2) * img_size] = random[i]

					if j == 0:
						result[i * img_size: (i + 1) * img_size, :img_size] = self.multi_to_one(label[i])

			imsave(os.path.join(self.args.result_dir, 'random.jpg'), result)

		elif self.args.test_mode == 'combine':
			img_paths = glob(os.path.join('dataset', self.args.dataset_name, self.args.test_img_dir, '*'))
			label_paths = glob(os.path.join('dataset', self.args.dataset_name, self.args.test_label_dir, '*'))
			label = np.array([load_label(label_path) for label_path in label_paths])

			result = np.zeros(((len(img_paths) + 1) * img_size, (len(label_paths) + 1) * img_size, self.args.img_nc))
			for j in range(len(img_paths)):
				img = imread(img_paths[j])
				result[:img_size, (j + 1) * img_size: (j + 2) * img_size] = img

				img = np.repeat(np.expand_dims(imnorm(img), 0), len(label_paths), 0)
				mean, var = self.E(img)
				random = self.G(label, mean, var, random_style=False)
				random = imdenorm(random.numpy())
				
				for i in range(len(label_paths)):
					result[(i + 1) * img_size: (i + 2) * img_size, (j + 1) * img_size: (j + 2) * img_size] = random[i]

					if j == 0:
						result[i * img_size: (i + 1) * img_size, :img_size] = self.multi_to_one(label[i])

			imsave(os.path.join(self.args.result_dir, 'combine.jpg'), result)

	def multi_to_one(self, data):
		return np.expand_dims(np.argmax(data, -1) / (self.args.label_nc - 1), -1)

	def load(self, all_module=False):
		self.E = tk.models.load_model(os.path.join(self.args.save_dir, 'E.h5'))
		self.G = decoder()
		self.G.load_weights(os.path.join(self.args.save_dir, 'G.h5'))

		with open(os.path.join(self.args.save_dir, 'sample.pkl'), 'rb') as fr:
			label = pickle.load(fr)
			self.G(label, None, None, random_style=True)
		
		if all_module:
			self.D = tk.models.load_model(os.path.join(self.args.save_dir, 'D.h5'))

	def save(self, label):
		self.E.save(os.path.join(self.args.save_dir, 'E.h5'))
		self.G.save_weights(os.path.join(self.args.save_dir, 'G.h5'))
		self.D.save(os.path.join(self.args.save_dir, 'D.h5'))
		
		with open(os.path.join(self.args.save_dir, 'sample.pkl'), 'wb') as fw:
			pickle.dump(label, fw)