# -*- coding: utf-8 -*-

import os, argparse
from dataloader import Dataloader
from model import Model
import sys
sys.path.append('..')
from utils import check_dir

def parse_args():
	desc = 'TensorFlow 2.0 implementation of Residual Attribute Generative Adversarial Network (RAG)'
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--dataset_name', type=str, default='celeba')
	parser.add_argument('--phase', type=str, default='tfrecord', choices=('tfrecord', 'train', 'test'))
	parser.add_argument('--img_size', type=int, default=128)
	parser.add_argument('--img_nc', type=int, default=3)

	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--epochs', type=int, default=10)
	parser.add_argument('--decay_epochs', type=int, default=10)
	parser.add_argument('--w_adv', type=float, default=1)
	parser.add_argument('--w_cls', type=float, default=10)
	parser.add_argument('--w_cyc', type=float, default=10)
	parser.add_argument('--w_rec', type=float, default=10)
	parser.add_argument('--w_a', type=float, default=1)
	parser.add_argument('--w_tv', type=float, default=2.5)
	parser.add_argument('--gan_type', type=str, default='lsgan', choices=('vanilla', 'lsgan', 'hinge'))

	parser.add_argument('--log_freq', type=int, default=1000)
	parser.add_argument('--output_dir', type=str, default='output')
	parser.add_argument('--log_dir', type=str, default='log')
	parser.add_argument('--sample_dir', type=str, default='sample')
	parser.add_argument('--save_dir', type=str, default='model')
	parser.add_argument('--result_dir', type=str, default='result')
	parser.add_argument('--test_img', type=str, default='000009.jpg')

	args = parser.parse_args()
	check_dir(args.output_dir)
	args.output_dir = os.path.join(args.output_dir, f'RAG_{args.dataset_name}')
	check_dir(args.output_dir)
	args.log_dir = os.path.join(args.output_dir, args.log_dir)
	check_dir(args.log_dir)
	args.sample_dir = os.path.join(args.output_dir, args.sample_dir)
	check_dir(args.sample_dir)
	args.save_dir = os.path.join(args.output_dir, args.save_dir)
	check_dir(args.save_dir)
	args.result_dir = os.path.join(args.output_dir, args.result_dir)
	check_dir(args.result_dir)

	if args.dataset_name == 'celeba':
		args.shorter_size = 178
		args.attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young', 'Eyeglasses', 
			'Mouth_Slightly_Open', 'Pale_Skin', 'Rosy_Cheeks', 'Smiling', 'Heavy_Makeup']
		args.label_nc = len(args.attrs)

	return args

if __name__ == '__main__':
	args = parse_args()
	if args.phase == 'tfrecord':
		print('Converting data to tfrecord...')
		Dataloader(args)
		print('Convert finished...')

	else:
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
