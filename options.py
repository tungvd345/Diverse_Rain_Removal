import argparse
import os
import torch
import datetime
from helper import *

class TrainOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser()
		self.initialized = False
		
	def initialize(self):
		# BaseOptions.initialize(self)
		# self.parser.add_argument('--data_dir', default='G:/DATASET/Heavy_rain_image_cvpr2019/CVPR19HeavyRainTrain/train', help='dataset directory')
		# self.parser.add_argument('--data_dir', default='D:/DATASETS/DID-MDN-datasets/DID-MDN-training/Rain_Heavy', help='dataset directory')
		self.parser.add_argument('--data_dir', default='D:/NewFolder/Diverse_Rain_dataset/train', help='dataset directory')
		self.parser.add_argument('--save_dir', default='./save', help='datasave directory')

		self.parser.add_argument('--val_data_dir', default='D:/NewFolder/Diverse_Rain_dataset/test', help='val data directory')
		self.parser.add_argument('--val_batch_size', type=int, default=1)

		self.parser.add_argument('--load', default='Deraining', help='save result')
		self.parser.add_argument('--model_name', default='Deraining', help='model to select')
		self.parser.add_argument('--finetuning', default=True, help='finetuning the training')
		self.parser.add_argument('--pretrained_model', default='save/Deraining/model/model_299.pt', help='save result - model to start finetune')

		self.parser.add_argument('--nchannel', type=int, default=3, help ='number of color channel to use')
		# self.parser.add_argument('--nkeypoint', type=int, default=128, help='number of keypoints to use')
		self.parser.add_argument('--loadSizeX', type=int, default=1024, help='scale images to this size')
		self.parser.add_argument('--loadSizeY', type=int, default=1024, help='scale images to this size')
		self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
		self.parser.add_argument('--need_patch', default=True, help='get the patch from image')
		self.parser.add_argument('--patch_size', type=int, default = 256, help='patch size')

		self.parser.add_argument('--nThreads', type=int, default=8, help='number of threads for data loading')
		self.parser.add_argument('--batch_size', type=int, default=4, help='input batch size for training')
		self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
		self.parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
		self.parser.add_argument('--lrDecay', type=int, default=100, help='epoch of half lr')
		self.parser.add_argument('--decayType', default='inv', help='lr decay function')
		self.parser.add_argument('--lossType', default='MSE', help='Loss type')

		self.parser.add_argument('--period', type=int, default=1, help='period of evaluation')
		self.parser.add_argument('--gpu', type=int, default=0, help='gpu index')
		self.parser.add_argument('--scale', type=int, default=2, help='scale output size /input size')

		self.parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
		self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
		self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
		self.parser.add_argument('--seed', type=int, default=345, help='random seed')

		self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
		self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
		self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
		self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
		self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
		self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
		self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
		self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
		self.parser.add_argument('--niter', type=int, default=150, help='# of iter at starting learning rate')
		self.parser.add_argument('--niter_decay', type=int, default=150, help='# of iter to linearly decay learning rate to zero')
		self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
		self.parser.add_argument('--lambda_A', type=float, default=100.0, help='weight for cycle loss (A -> B -> A)')
		self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
		self.parser.add_argument('--identity', type=float, default=0.0, help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
		self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
		self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
		self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop',
								 help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
		self.parser.add_argument('--no_flip', default=False, help='if specified, do not flip the images for data augmentation')
		self.isTrain = True

		self.initialized = True

	def parse(self):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()
		self.opt.isTrain = self.isTrain   # train or test

		str_ids = self.opt.gpu_ids.split(',')
		self.opt.gpu_ids = []
		for str_id in str_ids:
			id = int(str_id)
			if id >= 0:
				self.opt.gpu_ids.append(id)

		# set gpu ids
		if len(self.opt.gpu_ids) > 0:
			torch.cuda.set_device(self.opt.gpu_ids[0])

		args = vars(self.opt)
		'''
		print('------------ Options -------------')
		for k, v in sorted(args.items()):
			print('%s: %s' % (str(k), str(v)))
		print('-------------- End ----------------')
		'''

		# save to the disk
		expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
		mkdirs(expr_dir)
		file_name = os.path.join(expr_dir, 'opt.txt')
		# with open(file_name, 'wt') as opt_file:
		with open(file_name, 'a') as opt_file:
			date_time = str(datetime.datetime.now())
			opt_file.write('------------ Options -------------'+date_time+'\n')
			for k, v in sorted(args.items()):
				opt_file.write('%s: %s\n' % (str(k), str(v)))
			opt_file.write('-------------- End ----------------\n')
		return self.opt