import os
import argparse
import torch
import numpy as np
import torchvision.transforms as transforms
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from cal_ssim import SSIM

parser = argparse.ArgumentParser(description='Deraining')

# parser.add_argument('--data_dir_in', required=False, default='D:/rain_comparison/2.JCAS/results')
# parser.add_argument('--data_dir_in', required=False, default='D:/rain_comparison/6.RESCAN-master/showdir')
# parser.add_argument('--data_dir_in', required=False, default='D:/rain_comparison/7.DID-MDN-master/result_all/test_train_heavydata_epoch39_HR')
# parser.add_argument('--data_dir_in', required=False, default='D:/rain_comparison/8.SPANet-master/test_HR/pred')
# parser.add_argument('--data_dir_in', required=False, default='D:/rain_comparison/9.PReNet-master/results/PreNet_Rain1400_HR')
# parser.add_argument('--data_dir_in', required=False, default='D:/rain_comparison/10.UMRL--using-Cycle-Spinning-master/result_all/test_heavy_epoch49_HR')
# parser.add_argument('--data_dir_in', required=False, default='D:/rain_comparison/14.RCDNet-master/RCDNet_code/experiment/RCDNet_test/results')
# parser.add_argument('--data_dir_in', required=False, default='D:/rain_comparison/11.Semi-supervised-IRR-master/test_results_finetuned_epoch15')
# parser.add_argument('--data_dir_in', required=False, default='D:/rain_comparison/15.MSPFN-master/model/test/results_MSPFN_pretrained30')
# parser.add_argument('--data_dir_in', required=False, default='D:/rain_comparison/16.DGNL-Net/results_40k_heavyrain/predict_40k_heavyrain')
# parser.add_argument('--data_dir_in', required=False, default='D:/rain_comparison/17.DRD-Net-master/DRD-Net/result_Outdoor-Rain')
# parser.add_argument('--data_dir_in', required=False, default="D:/Deraining_TungVu/extract/out_img")
parser.add_argument('--data_dir_in', required=False, default='D:/rain_comparison/7.DID-MDN-master/result_DIDMDN_data')
# parser.add_argument('--data_dir_tar', required=False, default='D:/DATASETS/Heavy_rain_image_cvpr2019/test_with_train_param_v5/gt')
parser.add_argument('--data_dir_tar', required=False, default='C:/DATASETS/DID-MDN-datasets/DID-MDN_test_orig')

args = parser.parse_args()
for arg in vars(args):
    print (arg, getattr(args, arg))
# print(args)

def evaluate(args):
    path_in = args.data_dir_in
    path_tar = args.data_dir_tar
    file_in = sorted(os.listdir(path_in))
    file_tar = sorted(os.listdir(path_tar))
    len_list_in = len(file_in)

    # calculate PSNR, SSIM
    psnr_avg = 0
    ssim_avg = 0
    ssim_avg_self = 0
    # SSIM_func = SSIM().cuda()

    for i in range(len_list_in):
        list_in = os.path.join(path_in, file_in[i])
        # list_tar = os.path.join(path_tar, file_tar[i//15])
        list_tar = os.path.join(path_tar, file_tar[i])
        img_in = cv2.imread(list_in)
        img_tar = cv2.imread(list_tar)

        mse = ((img_in - img_tar) ** 2).mean()
        # psnr_tmp = 10 * log10(255 * 255 / (mse + 10 ** (-10)))
        # psnr_avg += psnr_tmp
        psnr_tmp = psnr(img_in, img_tar, data_range=255)
        psnr_avg += psnr_tmp

        ssim_tmp = ssim(img_in, img_tar, data_range=255, multichannel=True)
        ssim_avg += ssim_tmp

        # img_in_torch, img_tar_torch = RGB_np2tensor(img_in, img_tar)
        # c, h, w = img_in_torch.shape
        # img_in_torch = torch.reshape(img_in_torch, (1, c, h, w))
        # img_tar_torch = torch.reshape(img_tar_torch, (1, c, h, w))
        # ssim_tmp_self = SSIM_func(img_in_torch, img_tar_torch)
        # ssim_avg_self += ssim_tmp_self
        print('%s: PSNR = %2.5f, SSIM = %2.5f' % (file_in[i], psnr_tmp, ssim_tmp))

    psnr_avg = psnr_avg / len_list_in
    ssim_avg = ssim_avg / len_list_in
    # ssim_avg_self = ssim_avg_self / len_list_in
    print('avg psnr = %2.5f, avg SSIM = %1.5f' %(psnr_avg, ssim_avg))

if __name__ == '__main__':
    evaluate(args)