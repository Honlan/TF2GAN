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
	parser.add_argument('--save_freq', type=int, default=10000)
	parser.add_argument('--output_dir', type=str, default='output')
	parser.add_argument('--log_dir', type=str, default='log')
	parser.add_argument('--sample_dir', type=str, default='sample')
	parser.add_argument('--save_dir', type=str, default='model')
	parser.add_argument('--result_dir', type=str, default='result')

	parser.add_argument('--lr', type=float, default=0.0002)
	parser.add_argument('--gan_type', type=str, default='lsgan', choices=('vanilla', 'lsgan', 'hinge'))

	args = parser.parse_args()
	check_dir(args.output_dir)
	args.output_dir = os.path.join(args.output_dir, f'DCGAN_{args.dataset_name}')
	check_dir(args.output_dir)
	args.log_dir = os.path.join(args.output_dir, args.log_dir)
	check_dir(args.log_dir)
	args.sample_dir = os.path.join(args.output_dir, args.sample_dir)
	check_dir(args.sample_dir)
	args.save_dir = os.path.join(args.output_dir, args.save_dir)
	check_dir(args.save_dir)
	args.result_dir = os.path.join(args.output_dir, args.result_dir)
	check_dir(args.result_dir)

	return args

if __name__ == '__main__':
	args = parse_args()
	model = Model(args)
	model.build_model()

	if args.phase == 'train':
		print('Training...')
		model.train()
		print('Train finished...')
	
	elif args.phase == 'test':
		print('Testing...')
		model.test()
		print('Test finished...')