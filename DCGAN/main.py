# -*- coding: utf-8 -*-

import os, argparse
from model import Model
import sys
sys.path.append('..')
from utils import check_dir

def parse_args():
	desc = 'TensorFlow 2.0 implementation of Deep Convolutional Generative Adversarial Network (DCGAN)'
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--dataset_name', type=str, default='celeba')
	parser.add_argument('--phase', type=str, default='train', choices=('train', 'test'))
	parser.add_argument('--img_size', type=int, default=64)
	parser.add_argument('--img_nc', type=int, default=3)
	parser.add_argument('--z_dim', type=int, default=128)

	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--iteration', type=int, default=100000)
	parser.add_argument('--log_freq', type=int, default=1000)
	parser.add_argument('--sample_freq', type=int, default=1000)
	parser.add_argument('--checkpoint_freq', type=int, default=10000)
	parser.add_argument('--model_dir', type=str, default='models')
	parser.add_argument('--log_dir', type=str, default='logs')
	parser.add_argument('--sample_dir', type=str, default='samples')
	parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
	parser.add_argument('--result_dir', type=str, default='results')

	parser.add_argument('--lr', type=float, default=0.0002)
	parser.add_argument('--gan_type', type=str, default='hinge', choices=('vanilla', 'wgan', 'lsgan', 'hinge'))
	parser.add_argument('--w_gp', type=float, default=10.0)

	args = parser.parse_args()
	check_dir(args.model_dir)
	args.model_dir = os.path.join(args.model_dir, f'DCGAN_{args.dataset_name}')
	check_dir(args.model_dir)
	args.log_dir = os.path.join(args.model_dir, args.log_dir)
	check_dir(args.log_dir)
	args.sample_dir = os.path.join(args.model_dir, args.sample_dir)
	check_dir(args.sample_dir)
	args.checkpoint_dir = os.path.join(args.model_dir, args.checkpoint_dir)
	check_dir(args.checkpoint_dir)
	args.result_dir = os.path.join(args.model_dir, args.result_dir)
	check_dir(args.result_dir)

	return args

if __name__ == '__main__':
	args = parse_args()
	gan = Model(args)
	gan.build_model()

	if args.phase == 'train':
		print('Training...')
		model.train()
		print('Train finished...')
	elif args.phase == 'test':
		print('Testing...')
		model.test()
		print('Test finished...')