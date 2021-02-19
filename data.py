import os
import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import glob
from PIL import Image
import torchvision.transforms as transforms

def RGB_np2tensor(imgIn, imgTar, imgTarLR, channel):
    if channel == 1:
        # rgb --> Y (gray)
        imgIn = np.sum(imgIn * np.reshape([65.481, 128.553, 24.966], [1, 1, 3]) / 255.0, axis=2, keepdims=True) + 16.0
        imgTar = np.sum(imgTar * np.reshape([65.481, 128.553, 24.966], [1, 1, 3]) / 255.0, axis=2, keepdims=True) + 16.0
        imgTarLR = np.sum(imgTarLR * np.reshape([65.481, 128.553, 24.966], [1, 1, 3]) / 255.0, axis=2, keepdims=True) + 16.0

    # to Tensor
    ts = (2, 0, 1)
    imgIn = torch.Tensor(imgIn.transpose(ts).astype(float)).mul_(1.0)
    imgTar = torch.Tensor(imgTar.transpose(ts).astype(float)).mul_(1.0)
    imgTarLR = torch.Tensor(imgTarLR.transpose(ts).astype(float)).mul_(1.0)

    # normalization [-1,1]
    # imgIn = (imgIn/255.0 - 0.5) * 2
    # imgTar = (imgTar/255.0 - 0.5) * 2

    transform_list = [transforms.Normalize((0, 0, 0), (255, 255, 255)),]
                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    normalize = transforms.Compose(transform_list)
    imgIn = normalize(imgIn)
    imgTar = normalize(imgTar)
    imgTarLR = normalize(imgTarLR)

    return imgIn, imgTar, imgTarLR


def RGB_np2tensor_kpt(keypoints_in, num_keypoints):
    # to Tensor
    ts = (2, 0, 1)
    keypoints_in = torch.Tensor(keypoints_in.transpose(ts).astype(float)).mul_(1.0)

    # normalization
    transform_list = [transforms.Normalize((0, 0, 0)*num_keypoints, (255, 255, 255)*num_keypoints),]
                      # transforms.Normalize((0.5, 0.5, 0.5)*num_keypoints, (0.5, 0.5, 0.5)*num_keypoints)]
    normalize = transforms.Compose(transform_list)
    keypoints_in = normalize(keypoints_in)

    return keypoints_in

def getPatch(imgIn, imgTar, args):
    (ih, iw, c) = imgIn.shape
    patch_size_w = (args.patch_size)
    patch_size_h = args.patch_size # HR image patch size
    ix = random.randrange(0, iw - patch_size_w + 1)
    iy = random.randrange(0, ih - patch_size_h + 1)
    imgIn = imgIn[iy:iy + patch_size_h, ix:ix + patch_size_w, :]
    imgTar = imgTar[iy:iy + patch_size_h, ix:ix + patch_size_w, :]

    return imgIn, imgTar

def get_grid(imgIn, num_points):
    # imgIn = np.pad(imgIn, ([0, keypoint_size], [0, keypoint_size//2*3], [0,0]), 'constant',constant_values=0)
    h, w, c = imgIn.shape
    keypoint_size = ((h+num_points-1)//num_points, (w+num_points-1)//num_points)

    # nkeypoints = 128
    # pos = np.random.randint(h, size=(128, 2))

    imgIn = np.pad(imgIn, ([0, keypoint_size[0]], [0, keypoint_size[1]], [0,0]), 'edge')
    keypoints = np.zeros([keypoint_size[0], keypoint_size[1],3*num_points*num_points])
    for i in range(num_points):
        for j in range(num_points):
            idx = i*num_points + j
            pos = (i*keypoint_size[0], j*keypoint_size[1]) #(height, width)
            keypoints[:, :, idx*3:idx*3+3] = imgIn[pos[0] : pos[0]+keypoint_size[0], pos[1] : pos[1]+keypoint_size[1], :]

    return keypoints

def augment(imgIn, imgTar):
    if random.random() < 0.25: # horizontal flip
        imgIn = imgIn[:, ::-1, :]
        imgTar = imgTar[:, ::-1, :]
    elif random.random() < 0.5: # vertical flip
        imgIn = imgIn[::-1, :, :]
        imgTar = imgTar[::-1, :, :]
    elif random.random() < 0.75: # horizontal + vertical filp
        imgIn = imgIn[::-1, ::-1, :]
        imgTar = imgTar[::-1, ::-1, :]
    else:
        imgIn= imgIn
        imgTar = imgTar

    # rot = random.randint(0, 1) # rotate 0/180 degree
    # imgIn = np.rot90(imgIn, rot*2, (0, 1))
    # imgTar = np.rot90(imgTar, rot*2, (0, 1))

    return imgIn, imgTar

def make_dataset(dir):
    images = []
    file_name = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)
            file_name.append(fname)
    return sorted(images), sorted(file_name)

class outdoor_rain_train(data.Dataset):
    def __init__(self, args):
        self.args = args
        # self.crop = transforms.RandomCrop(args.patch_size)
        apath = args.data_dir

        ''' read data base on os.path
        dir_rainy = 'rain'
        dir_clear = 'norain'
        self.dir_in = os.path.join(apath, dir_rainy)
        self.dir_tar = os.path.join(apath, dir_clear)
        '''

        ''' read data base on glob - in multi file
        # dir_rainy = 'rain/*.png'
        # dir_clear = 'norain/*.png'
        # self.file_in_list = sorted(glob.glob(self.dir_in))
        # self.file_tar_list = sorted(glob.glob(self.dir_tar))
        '''

        # self.file_in_list = sorted(make_dataset(self.dir_in), key=lambda x: x[:-6])
        # self.file_tar_list = sorted(make_dataset(self.dir_tar))

        # self.dir = args.data_dir
        dir_rainy = 'medium'
        dir_clear = 'gt'
        self.dir_in = os.path.join(apath, dir_rainy)
        self.dir_tar = os.path.join(apath, dir_clear)

        self.file_in_list, self.file_in_name = sorted(make_dataset(self.dir_in))
        self.file_tar_list, self.file_tar_name = sorted(make_dataset(self.dir_tar))
        # self.transform = get_transform(args)
        self.len = len(self.file_in_list)
        ###############

    def __getitem__(self, idx):
        args = self.args
        img_in_LR = cv2.imread(self.file_in_list[idx])
        img_tar_LR = cv2.imread(self.file_tar_list[idx//5])
        if args.need_patch:
            img_in_LR, img_tar_LR = getPatch(img_in_LR, img_tar_LR, self.args)
        img_in_LR, img_tar_LR = augment(img_in_LR, img_tar_LR)
        # img_in_LR = img_in[::2, ::2, :]
        # img_tar_LR = img_tar[::2, ::2, :]

        #################################################
        num_p = 16
        keypoints_in = get_grid(img_in_LR, num_p)
        #################################################
        img_tar_LR, img_tar_LR, img_in_LR = RGB_np2tensor(img_tar_LR, img_tar_LR, img_in_LR, args.nchannel)
        keypoints_in = RGB_np2tensor_kpt(keypoints_in, num_p*num_p)
        return img_in_LR, keypoints_in, img_tar_LR, img_tar_LR

        #################### read dataset in JORDER
        # img_in = Image.open(self.file_in_list[idx]).convert("RGB")
        # img_tar = Image.open(self.file_tar_list[idx]).convert("RGB")
        # img_in = self.transform(img_in)
        # img_tar = self.transform(img_tar)
        ################################################

        ################## read dataset in DID-MDN
        # img_in_tar = Image.open(self.file_list[idx]).convert("RGB")
        # img_in_tar = self.transform(img_in_tar)
        # c, h, w = img_in_tar.size()
        # img_in = img_in_tar[:, :, :w//2]
        # img_tar = img_in_tar[:, :, w//2:]
        # if args.need_patch:
        #     img_in, img_tar = getPatch(img_in, img_tar, args)
        ###############################################################

        # #####################read data heavy rain
        # img_in = Image.open(self.file_in_list[idx]).convert("RGB")
        # img_tar = Image.open(self.file_tar_list[idx // 15]).convert("RGB")
        #
        # img_in = self.transform(img_in)
        # img_tar = self.transform(img_tar)
        # if args.need_patch:
        #     img_in, img_tar = getPatch(img_in, img_tar, args)
        # ###############################################################
        # return img_in, img_tar


    def __len__(self):
        return self.len

    def get_file_name(self, idx):
        name = self.fileList[idx]
        nameTar = os.path.join(self.dirTar, name)
        nameIn = os.path.join(self.dirIn, name)
        return nameIn, nameTar

class outdoor_rain_test(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.channel = args.nchannel
        apath = args.val_data_dir

        # dir_rainy = 'rain/X2'
        # dir_clear = 'norain'
        # self.dir_in = os.path.join(apath, dir_rainy)
        # self.dir_tar = os.path.join(apath, dir_clear)
        # self.file_in_list = sorted(make_dataset(self.dir_in), key=lambda x: x[:-6])
        # self.file_tar_list = sorted(make_dataset(self.dir_tar))

        dir_rainy = 'medium'
        dir_clear = 'gt'
        self.dir_in = os.path.join(apath, dir_rainy)
        self.dir_tar = os.path.join(apath, dir_clear)
        self.file_in_list, self.file_in_name = sorted(make_dataset(self.dir_in))
        self.file_tar_list, self.file_tar_name = sorted(make_dataset(self.dir_tar))

        # self.file_list = sorted(make_dataset(apath))
        # self.transform = transforms.Compose([transforms.ToTensor(),
        #                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.len = len(self.file_in_list)

    def __getitem__(self, idx):
        args = self.args
        #####################read data in DID-MDN
        # img_in_tar = Image.open(self.file_list[idx]).convert('RGB')
        # img_in_tar = self.transform(img_in_tar)
        # c, h, w = img_in_tar.size()
        # img_in = img_in_tar[:, :, :w // 2]
        # img_tar = img_in_tar[:, :, w // 2:]
        #################################################################

        ##################### use PIL image
        # img_in = Image.open(self.file_in_list[idx]).convert("RGB")
        # img_tar = Image.open(self.file_tar_list[idx // 15]).convert("RGB")
        # img_in = self.transform(img_in)
        # img_tar = self.transform(img_tar)
        ##################################################################

        ##################### use cv2 image
        img_in = cv2.imread(self.file_in_list[idx])
        # img_in_LR = img_in[::2, ::2, :]
        img_in_LR_name = self.file_in_name[idx]
        img_tar = cv2.imread(self.file_tar_list[idx // 5])
        # img_tar = cv2.imread(self.file_tar_name[idx])
        # img_tar_LR = img_tar[::2, ::2, :]
        img_tar_LR_name = self.file_tar_name[idx // 5]
        # img_tar_LR_name = self.file_tar_name[idx]

        num_p = 16
        keypoints_in = get_grid(img_in, num_p)

        img_tar_LR, img_tar, img_in_LR = RGB_np2tensor(img_tar, img_tar, img_in, args.nchannel)
        keypoints_in = RGB_np2tensor_kpt(keypoints_in, num_p*num_p)
        ##################################################################
        return img_in_LR, keypoints_in, img_tar_LR, img_tar, img_in_LR_name

    def __len__(self):
        return self.len

    def get_file_name(self, idx):
        name = self.fileList[idx]
        nameTar = os.path.join(self.dirTar, name)
        nameIn = os.path.join(self.dirIn, name)
        return nameIn, nameTar

class real_rain_test(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.dir_in = args.real_rain_data_dir
        # self.file_list, self.file_name = sorted(make_dataset(self.dir_in))
        self.file_name = sorted(os.listdir(self.dir_in))
        self.len = len(self.file_name)

    def __getitem__(self, idx):
        args = self.args

        ##################### use cv2 image
        real_dir = os.path.join(self.dir_in, self.file_name[idx])
        # img_in = cv2.imread(self.file_list[idx])
        img_in = cv2.imread(real_dir)
        img_in_LR_name = self.file_name[idx]

        num_p = 16
        keypoints_in = get_grid( img_in, num_p)

        _, _, img_in_LR = RGB_np2tensor(img_in, img_in, img_in, args.nchannel)
        keypoints_in = RGB_np2tensor_kpt(keypoints_in, num_p*num_p)
        ##################################################################
        return img_in_LR, keypoints_in, img_in_LR_name

    def __len__(self):
        return self.len

def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        # osize = [opt.loadSizeX, opt.loadSizeY]
        osize = [opt.fineSize, opt.fineSize]
        # transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        # transform_list.append(transforms.RandomCrop(opt.patch_size))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop((opt.patch_size, opt.patch_size//2*3)))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSizeX)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    if opt.isTrain and not opt.no_flip:
        print("augment is ok")
        if random.random() < 0.3:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif random.random() < 0.6:
            transform_list.append(transforms.RandomVerticalFlip())
        else:
            transform_list.append(transforms.RandomRotation((0,360)))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]

    return transforms.Compose(transform_list)

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)
