# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as tk

# Activations

def relu(x):
	return tf.nn.relu(x)

def Relu():
	return tk.layers.ReLU()

def lrelu(x, alpha=0.2):
	return tf.nn.leaky_relu(x, alpha)

def Lrelu(alpha=0.2):
	return tk.layers.LeakyReLU(alpha)

def sigmoid(x):
	return tf.nn.sigmoid(x)

def tanh(x):
	return tf.nn.tanh(x)

# Layers

def Input(input_shape):
	return tk.layers.Input(input_shape)

def Reshape(target_shape):
	return tk.layers.Reshape(target_shape)

def flatten(x):
	return tf.reshape(x, [x.shape[0], -1])

def Flatten():
	return tk.layers.Flatten()

def Add():
	return tk.layers.Add()

def Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'):
	return tk.layers.Dense(units=units, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)

def Conv2d(filters, kernel_size, strides, padding='same', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'):
	return tk.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)

def Deconv2d(filters, kernel_size, strides=2, padding='same', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'):
	return tk.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)

def Dropout(rate):
	return tk.layers.Dropout(rate)

class Resblock(tk.Model):
	def __init__(self, filters=256, kernel_size=3, norm='in', dropout=False, dropout_rate=0.5):
		super(Resblock, self).__init__()
		self.conv1 = Conv2d(filters, kernel_size, 1)
		self.conv2 = Conv2d(filters, kernel_size, 1)

		if norm == 'bn':
			self.norm1 = BN()
			self.norm2 = BN()
		elif norm == 'in':
			self.norm1 = IN()
			self.norm2 = IN()

		self.dropout = dropout
		if self.dropout:
			self.drop = Dropout(dropout_rate)

	def call(self, x):
		h = relu(self.norm1(self.conv1(x)))
		
		if self.dropout:
			h = self.drop(h)

		return x + self.norm2(self.conv2(h))

# Normalizations

def BN():
	return tk.layers.BatchNormalization()

class IN(tk.layers.Layer):
	def __init__(self, epsilon=1e-5):
		super(IN, self).__init__()
		self.epsilon = epsilon

	def build(self, input_shape):
		self.scale = self.add_weight(name='scale', shape=input_shape[-1:], initializer=tf.random_normal_initializer(1., 0.02), trainable=True)
		self.offset = self.add_weight(name='offset', shape=input_shape[-1:], initializer='zeros', trainable=True)

	def call(self, x):
		mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
		inv = tf.math.rsqrt(variance + self.epsilon)
		normalized = (x - mean) * inv
		return self.scale * normalized + self.offset

class AdaIN(tk.layers.Layer):
	def __init__(self, scale, offset, epsilon=1e-5):
		super(AdaIN, self).__init__()
		self.scale = scale
		self.offset = offset
		self.epsilon = epsilon

	def build(self, input_shape):
		pass

	def call(self, x):
		mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
		inv = tf.math.rsqrt(variance + self.epsilon)
		normalized = (x - mean) * inv
		return self.scale * normalized + self.offset

# Losses

def l1_loss(x, y):
	return tf.reduce_mean(tf.abs(x - y))

def l2_loss(x, y):
	return tf.reduce_mean(tf.square(x - y))

def discriminator_loss(real, fake, gan_type, multi_scale=False):
	if not multi_scale:
		real = [real]
		fake = [fake]

	cross_entropy = tk.losses.BinaryCrossentropy(from_logits=True)
	loss = []
	for i in range(len(fake)):
		if gan_type == 'vanilla':
			loss_real = cross_entropy(tf.ones_like(real[i]), real[i])
			loss_fake = cross_entropy(tf.zeros_like(fake[i]), fake[i])
		elif gan_type == 'lsgan':
			loss_real = tf.reduce_mean(tf.square(real[i] - 1.0))
			loss_fake = tf.reduce_mean(tf.square(fake[i]))
		elif gan_type == 'hinge':
			loss_real = tf.reduce_mean(relu(1.0 - real[i]))
			loss_fake = tf.reduce_mean(relu(1.0 + fake[i]))
		elif gan_type == 'wgan':
			loss_real = -tf.reduce_mean(real[i])
			loss_fake = tf.reduce_mean(fake[i])
		loss.append(loss_real + loss_fake)

	return tf.reduce_mean(loss)

def generator_loss(fake, gan_type, multi_scale=False):
	if not multi_scale:
		fake = [fake]

	cross_entropy = tk.losses.BinaryCrossentropy(from_logits=True)
	loss = []
	for i in range(len(fake)):
		if gan_type == 'vanilla':
			loss_fake = cross_entropy(tf.ones_like(fake[i]), fake[i])
		elif gan_type == 'lsgan':
			loss_fake = tf.reduce_mean(tf.square(fake[i] - 1.0))
		elif gan_type == 'hinge':
			loss_fake = -tf.reduce_mean(fake[i])
		elif gan_type == 'wgan':
			loss_fake = -tf.reduce_mean(fake[i])
		loss.append(loss_fake)

	return tf.reduce_mean(loss)

def gradient_penalty(D, inter_sample, w_gp=10):
	with tf.GradientTape() as tape:
		tape.watch(inter_sample)
		inter_logit = D(inter_sample, training=True)
	
	grad = tape.gradient(inter_logit, inter_sample)[0]
	norm = tf.norm(flatten(grad), axis=1)
	
	return w_gp * tf.reduce_mean(tf.square(norm - 1.))

# Others

def lerp_tf(start, end, ratio):
	return start + (end - start) * tf.clip_by_value(ratio, 0.0, 1.0)

def get_pixel_value(img, x, y): # img: N, H, W, C; x: N, H, W; y: N, H, W
	N, H, W = x.shape
	return tf.gather_nd(img, tf.stack([tf.tile(tf.reshape(tf.range(0, N), (N, 1, 1)), (1, H, W)), y, x], 3))

def grid_sample(img, x, y):
	_, H, W, _ = img.shape
	max_y = tf.cast(H - 1, 'int32')
	max_x = tf.cast(W - 1, 'int32')

	x = 0.5 * ((tf.cast(x, 'float32') + 1.0) * tf.cast(max_x - 1, 'float32'))
	y = 0.5 * ((tf.cast(y, 'float32') + 1.0) * tf.cast(max_y - 1, 'float32'))

	x0 = tf.clip_by_value(tf.cast(tf.floor(x), 'int32'), 0, max_x)
	x1 = tf.clip_by_value(x0 + 1, 0, max_x)
	y0 = tf.clip_by_value(tf.cast(tf.floor(y), 'int32'), 0, max_y)
	y1 = tf.clip_by_value(y0 + 1, 0, max_y)

	Ia = get_pixel_value(img, x0, y0)
	Ib = get_pixel_value(img, x0, y1)
	Ic = get_pixel_value(img, x1, y0)
	Id = get_pixel_value(img, x1, y1)

	x0 = tf.cast(x0, 'float32')
	x1 = tf.cast(x1, 'float32')
	y0 = tf.cast(y0, 'float32')
	y1 = tf.cast(y1, 'float32')

	wa = tf.expand_dims((x1 - x) * (y1 - y), -1)
	wb = tf.expand_dims((x1 - x) * (y - y0), -1)
	wc = tf.expand_dims((x - x0) * (y1 - y), -1)
	wd = tf.expand_dims((x - x0) * (y - y0), -1)

	return tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

def gram_matrix(x):
	N, H, W, C = x.shape
	x = tf.reshape(x, (N, H * W, C))
	return tf.matmul(tf.transpose(x, (0, 2, 1)), x) / (H * W * C)

def gram_mse(x, y):
	return tf.reduce_mean(tf.square(gram_matrix(x) - gram_matrix(y)))

def roi_align(features, boxes, indices, img_size, output_size):
	output_size[0] = output_size[0] * 2
	output_size[1] = output_size[1] * 2

	x0, y0, x1, y1 = tf.split(boxes, 4, axis=-1)
	binW = (x1 - x0) / output_size[1]
	binH = (y1 - y0) / output_size[0]

	nx0 = (x0 + binW / 2 - 0.5) / (img_size[1] - 1)
	ny0 = (y0 + binH / 2 - 0.5) / (img_size[0] - 1)
	nW = binW * (output_size[1] - 1) / (img_size[1] - 1)
	nH = binH * (output_size[0] - 1) / (img_size[0] - 1)

	new_boxes = tf.concat([ny0, nx0, ny0 + nH, nx0 + nW], axis=-1)
	sampled = tf.image.crop_and_resize(features, new_boxes, indices, output_size)

	return tf.nn.avg_pool2d(sampled, 2, 2, padding='VALID')