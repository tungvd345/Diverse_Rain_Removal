import os
import os.path
import torch
import sys
from torchvision import models
from collections import namedtuple

class saveData():
    def __init__(self, args):
        self.args = args
        self.save_dir = os.path.join(args.save_dir, args.load)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_dir_model = os.path.join(self.save_dir, 'model')
        if not os.path.exists(self.save_dir_model):
            os.makedirs(self.save_dir_model)
        if os.path.exists(self.save_dir + '/log.txt'):
            self.logFile = open(self.save_dir + '/log.txt', 'a')
        else:
            self.logFile = open(self.save_dir + '/log.txt', 'w')

    def save_model(self, model, epoch):
        torch.save(
            model.state_dict(),
            self.save_dir_model + '/model_lastest.pt')
        torch.save(
            model.state_dict(),
            self.save_dir_model + '/model_' + str(epoch) + '.pt')
        torch.save(
            model,
            self.save_dir_model + '/model_obj.pt')
        torch.save(
            epoch,
            self.save_dir_model + '/last_epoch.pt')

    def save_log(self, log):
        sys.stdout.flush()
        self.logFile.write(log + '\n')
        self.logFile.flush()

    def load_model(self, model):
        model.load_state_dict(torch.load(self.save_dir_model + '/model_lastest.pt'))
        last_epoch = torch.load(self.save_dir_model + '/last_epoch.pt')
        print("load mode_status from {}/model_lastest.pt, epoch: {}".format(self.save_dir_model, last_epoch))
        return model, last_epoch

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

def mkdirs(paths):
	if isinstance(paths, list) and not isinstance(paths, str):
		for path in paths:
			mkdir(path)
	else:
		mkdir(paths)

def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)

def gradient(x):
    gradient_h = torch.abs(x[:,:,:,:-1] - x[:,:,:,1:])
    gradient_v = torch.abs(x[:,:,:-1,:] - x[:,:,1:,:])
    return gradient_h, gradient_v


class ToTensor(object):
    """ Conver ndarray to Tensors"""
    def __call__(self, image_list):
        # input image_list is: H x W x C
        # torch image_list is: C x H x W
        tensor_list = []
        for image in image_list:
            image = image.transpose((2, 0, 1))
            tensor_list.append(image)
        return tensor_list


def tensor_to_image(tensor):
    if type(tensor) in [torch.autograd.Variable]:
        img = tensor.data[0].cpu().detach().numpy()
    else:
        img = tensor[0].cpu().detach().numpy()
    img = img.transpose((1,2,0))
    try:
        img = np.clip(img, 0, 255)
        if img.shape[-1] == 1:
            img = np.dstack((img, img, img))
    except:
        print("invalid value catch")
        Image.fromarray(img).save('catch.jpg')
    return img


def to_tensor(x, gpuid=None):
    if type(x) in [list, tuple]:
        image_num = len(x)
        if image_num >0:
            (h,w,c) = x[0].shape
        else:
            print("No image!")
        t = torch.FloatTensor(image_num, c, h, w)
        for i in range(image_num):
            image = x[i].transpose((2, 0, 1))
            t[i,:,:,:] = torch.from_numpy(image)
        if gpuid:
            t = t.cuda(gpuid)
        return t
    elif isinstance(x, np.ndarray):
        if len(x.shape) == 3:
            x = np.expand_dims(x, axis=0)
        elif len(x.shape) == 2:
            x = np.dstack((x,x,x))
            x = np.expand_dims(x, axis=0)
        bs, h, w, c = x.shape
        t = torch.FloatTensor(bs,c,h,w)
        x = x.transpose((0,3,1,2))
        t = torch.from_numpy(x)
        if gpuid:
            t = t.cuda(gpuid)
        return t
    else:
        print("data type not accepted!")
        return None


def to_variable(x, gpuid=3):
    v = None
    if type(x) in [list, tuple,  np.ndarray]:
        x = to_tensor(x)
    if type(x) in [torch.DoubleTensor, torch.FloatTensor]:
        if gpuid:
            x = x.cuda(gpuid)
        v = torch.autograd.Variable(x)
    else:
        print("Unrecognized data type!")
    return v


def generate_new_seq(filename):  # new function
    file_list = sorted(glob.glob(filename))
    return file_list  # [1:10000]


# def augment(input_list, scale_limit=300, crop_size=224):
#     input_list = RandomHorizontalFlip(input_list)
#     input_list = RandomColorWarp(input_list)
#     # input_list = RandomScale(rain, streak, clean, size_limit=scale_limit)
#     input_list = RandomCrop(input_list, size=crop_size)
#     return input_list


def compute_psnr(est, gt):
    batch_size = est.size()[0]
    sum_acc = 0
    for i in range(batch_size):
        est_image = est.cpu().data[i].detach().numpy()
        gt_image = gt.cpu().data[i].detach().numpy()
        est_image = est_image.transpose((1,2,0))
        gt_image = gt_image.transpose((1,2,0))
        sum_acc += psnr(est_image*255, gt_image*255)
    avg_acc = sum_acc / batch_size
    return avg_acc


def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)


def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    return np.float64(rgb.dot(xform.T))


def psnr(es, gt):
    if len(es.shape) ==3 and es.shape[2] == 3:
        es_img = rgb2ycbcr(es)
        gt_img = rgb2ycbcr(gt)
        es_channel = es_img[:,:,0]
        gt_channel = gt_img[:,:,0]
    else:
        es_channel = es
        gt_channel = gt

    imdiff = np.float64(es_channel) - np.float64(gt_channel)
    rmse = np.sqrt(np.mean(np.square(imdiff.flatten())))
    psnr_value = 20*np.log10(255/rmse)
    return psnr_value


def load_checkpoint(self, best=False):
    """
    Load the best copy of a model. This is useful for 2 cases:

    - Resuming training with the most recent model checkpoint.
    - Loading the best validation model to evaluate on the test data.

    Params
    ------
    - best: if set to True, loads the best model. Use this if you want
      to evaluate your model on the test data. Else, set to False in
      which case the most recent version of the checkpoint is used.
    """
    print("[*] Loading model from {}".format(self.ckpt_dir))

    filename = self.model_name + '_ckpt.pth.tar'
    if best:
        filename = self.model_name + '_model_best.pth.tar'
    ckpt_path = os.path.join(self.ckpt_dir, filename)
    ckpt = torch.load(ckpt_path)

    # load variables from checkpoint
    self.start_epoch = ckpt['epoch']
    self.best_valid_acc = ckpt['best_valid_acc']
    self.lr = ckpt['lr']
    self.model.load_state_dict(ckpt['state_dict'])

    if best:
        print(
            "[*] Loaded {} checkpoint @ epoch {} "
            "with best valid acc of {:.3f}".format(
                filename, ckpt['epoch']+1, ckpt['best_valid_acc'])
        )
    else:
        print(
            "[*] Loaded {} checkpoint @ epoch {}".format(
                filename, ckpt['epoch']+1)
        )
