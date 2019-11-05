# -*- coding: utf-8 -*-

import os, argparse
from model import Model
import sys
sys.path.append('..')
from utils import check_dir

def parse_args():
	desc = 'TensorFlow 2.0 implementation of SinGAN'
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--phase', type=str, default='train', choices=('train', 'test'))
	parser.add_argument('--input_image', type=str, default='balloons.png')
	parser.add_argument('--img_nc', type=int, default=3)
	parser.add_argument('--max_size', type=int, default=256)
	parser.add_argument('--min_size', type=int, default=25)

	parser.add_argument('--num_filter', type=int, default=32)
	parser.add_argument('--scale_factor', type=float, default=0.75)
	parser.add_argument('--noise_weight', type=float, default=0.1)

	parser.add_argument('--iteration', type=int, default=2000)
	parser.add_argument('--lr', type=float, default=0.0005)
	parser.add_argument('--decay_steps', type=float, default=1600)
	parser.add_argument('--decay_rate', type=float, default=0.1)
	
	parser.add_argument('--gan_type', type=str, default='wgan', choices=('vanilla', 'wgan', 'lsgan', 'hinge'))
	parser.add_argument('--w_gp', type=float, default=0.1)
	parser.add_argument('--w_rec', type=float, default=10)
	parser.add_argument('--G_step', type=int, default=3)
	parser.add_argument('--D_step', type=int, default=3)
	parser.add_argument('--output_dir', type=str, default='output')
	parser.add_argument('--save_dir', type=str, default='model')
	parser.add_argument('--result_dir', type=str, default='result')

	args = parser.parse_args()
	check_dir(args.output_dir)
	args.output_dir = os.path.join(args.output_dir, f'SinGAN_{args.input_image.split('.')[0]}')
	check_dir(args.output_dir)
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