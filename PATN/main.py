# -*- coding: utf-8 -*-

import os, argparse
from dataloader import Dataloader
from model import Model
import sys
sys.path.append('..')
from utils import check_dir

def parse_args():
	desc = 'TensorFlow 2.0 implementation of Progressive Pose Attention Transfer Network (PATN)'
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--dataset_name', type=str, default='fashion')
	parser.add_argument('--phase', type=str, default='tfrecord', choices=('tfrecord', 'train', 'test'))
	parser.add_argument('--img_h', type=int, default=256)
	parser.add_argument('--img_w', type=int, default=176)
	parser.add_argument('--img_nc', type=int, default=3)

	parser.add_argument('--batch_size', type=int, default=4)
	parser.add_argument('--train_pairs_csv', type=str, default='train_pairs.csv')
	parser.add_argument('--test_pairs_csv', type=str, default='test_pairs.csv')
	parser.add_argument('--train_kps_csv', type=str, default='train_kps.csv')
	parser.add_argument('--test_kps_csv', type=str, default='test_kps.csv')
	parser.add_argument('--kps_bbox_pkl', type=str, default='kps_bbox.pkl')
	parser.add_argument('--n_kps', type=int, default=18)
	parser.add_argument('--n_body_part', type=int, default=7)
	parser.add_argument('--n_attn_block', type=int, default=6)

	parser.add_argument('--iteration', type=int, default=3000)
	parser.add_argument('--epochs', type=int, default=20)
	parser.add_argument('--decay_epochs', type=int, default=10)
	parser.add_argument('--lr', type=float, default=0.0002)
	parser.add_argument('--gan_type', type=str, default='hinge', choices=('vanilla', 'lsgan', 'hinge'))
	parser.add_argument('--w_adv', type=float, default=5)
	parser.add_argument('--w_rec', type=float, default=1)
	parser.add_argument('--w_per', type=float, default=1)
	parser.add_argument('--w_style', type=float, default=10)
	parser.add_argument('--vgg_layer', type=str, default='block1_conv2', choices=('block1_conv2', 'block2_conv2', 'block3_conv4'))

	parser.add_argument('--log_freq', type=int, default=1000)
	parser.add_argument('--output_dir', type=str, default='output')
	parser.add_argument('--log_dir', type=str, default='log')
	parser.add_argument('--sample_dir', type=str, default='sample')
	parser.add_argument('--save_dir', type=str, default='model')
	parser.add_argument('--result_dir', type=str, default='result')

	args = parser.parse_args()
	check_dir(args.output_dir)
	args.output_dir = os.path.join(args.output_dir, f'PATN_{args.dataset_name}')
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