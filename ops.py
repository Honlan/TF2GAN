# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as tk

# Activations

def relu(x):
	return tf.nn.relu(x)

def lrelu(x, alpha=0.2):
	return tf.nn.leaky_relu(x, alpha=alpha)

def sigmoid(x):
	return tf.nn.sigmoid(x)

def tanh(x):
	return tf.nn.tanh(x)

def Relu():
	return tk.layers.ReLU()

def Lrelu(alpha=0.2):
	return tk.layers.LeakyReLU(alpha=alpha)

# Layers

def reshape(target_shape):
	return tk.layers.Reshape(target_shape)

def flatten():
	return tk.layers.Flatten()

def dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'):
	return tk.layers.Dense(units=units, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)

def conv2d(filters, kernel_size, strides, padding='same', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'):
	return tk.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)

def deconv2d(filters, kernel_size, strides=2, padding='same', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'):
	return tk.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)

# Normalizations

def batch_norm():
	return tk.layers.BatchNormalization()

class InstanceNormalization(tf.layers.Layer):
	def __init__(self, epsilon=1e-5):
		super(InstanceNormalization, self).__init__()
		self.epsilon = epsilon

	def build(self, input_shape):
		self.scale = self.add_weight(name='scale', shape=input_shape[-1:], initializer=tf.random_normal_initializer(1., 0.02), trainable=True)
		self.offset = self.add_weight(name='offset', shape=input_shape[-1:], initializer='zeros', trainable=True)

	def call(self, x):
		mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
		inv = tf.math.rsqrt(variance + self.epsilon)
		normalized = (x - mean) * inv
		return self.scale * normalized + self.offset

def instance_norm():
	return InstanceNormalization()

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
		elif gan_type == 'wgan':
			loss_real = -tf.reduce_mean(real[i])
			loss_fake = tf.reduce_mean(fake[i])
		elif gan_type == 'lsgan':
			loss_real = tf.reduce_mean(tf.square(real[i] - 1.0))
			loss_fake = tf.reduce_mean(tf.square(fake[i]))
		elif gan_type == 'hinge':
			loss_real = tf.reduce_mean(relu(1.0 - real[i]))
			loss_fake = tf.reduce_mean(relu(1.0 + fake[i]))

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
		elif gan_type == 'wgan':
			loss_fake = -tf.reduce_mean(fake[i])
		elif gan_type == 'lsgan':
			loss_fake = tf.reduce_mean(tf.square(fake[i] - 1.0))
		elif gan_type == 'hinge':
			loss_fake = -tf.reduce_mean(fake[i])

		loss.append(loss_fake)

	return tf.reduce_mean(loss)

