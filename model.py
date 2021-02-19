import torch
import torch.nn as nn
from torch.nn import init
import torchvision.transforms as transforms
import torchvision.models as models
import functools
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.res_blocks = 8
        self.conv_input = ConvLayer(3, 16, kernel_size=11, stride=1)
        self.dense0 = nn.Sequential(
            RCAB(16),
            RCAB(16)
        )

        self.conv2x = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.fusion1 = Encoder_MDCBlock1(32, 2, mode='iter2')
        self.dense1 = nn.Sequential(
            RCAB(32),
            RCAB(32)
        )

        self.conv4x = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.fusion2 = Encoder_MDCBlock1(64, 3, mode='iter2')
        self.dense2 = nn.Sequential(
            RCAB(64),
            RCAB(64)
        )

        self.conv8x = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.fusion3 = Encoder_MDCBlock1(128, 4, mode='iter2')
        self.dense3 = nn.Sequential(
            RCAB(128),
            RCAB(128)
        )

        self.conv16x = ConvLayer(128, 256, kernel_size=3, stride=2)
        self.fusion4 = Encoder_MDCBlock1(256, 5, mode='iter2')
        #self.dense4 = Dense_Block(256, 256)

        self.dehaze = nn.Sequential()
        for i in range(0, self.res_blocks):
            self.dehaze.add_module('res%d' % i, RCAB(256))

        self.convd16x = UpsampleConvLayer(256, 128, kernel_size=3, stride=2)
        self.dense_4 = nn.Sequential(
            RCAB(128),
            RCAB(128)
        )
        self.fusion_4 = Decoder_MDCBlock1(128, 2, mode='iter2')

        self.convd8x = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)
        self.dense_3 = nn.Sequential(
            RCAB(64),
            RCAB(64)
        )
        self.fusion_3 = Decoder_MDCBlock1(64, 3, mode='iter2')

        self.convd4x = UpsampleConvLayer(64, 32, kernel_size=3, stride=2)
        self.dense_2 = nn.Sequential(
            RCAB(32),
            RCAB(32)
        )
        self.fusion_2 = Decoder_MDCBlock1(32, 4, mode='iter2')

        self.convd2x = UpsampleConvLayer(32, 16, kernel_size=3, stride=2)
        self.dense_1 = nn.Sequential(
            RCAB(16),
            RCAB(16)
        )
        self.fusion_1 = Decoder_MDCBlock1(16, 5, mode='iter2')

        self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1)


    def forward(self, x):
        res1x = self.conv_input(x)
        feature_mem = [res1x]
        x = self.dense0(res1x) + res1x

        res2x = self.conv2x(x)
        res2x = self.fusion1(res2x, feature_mem)
        feature_mem.append(res2x)
        res2x =self.dense1(res2x) + res2x

        res4x =self.conv4x(res2x)
        res4x = self.fusion2(res4x, feature_mem)
        feature_mem.append(res4x)
        res4x = self.dense2(res4x) + res4x

        res8x = self.conv8x(res4x)
        res8x = self.fusion3(res8x, feature_mem)
        feature_mem.append(res8x)
        res8x = self.dense3(res8x) + res8x

        res16x = self.conv16x(res8x)
        res16x = self.fusion4(res16x, feature_mem)
        #res16x = self.dense4(res16x)

        res_dehaze = res16x
        in_ft = res16x*2
        res16x = self.dehaze(in_ft) + in_ft - res_dehaze
        feature_mem_up = [res16x]

        res16x = self.convd16x(res16x)
        res16x = F.upsample(res16x, res8x.size()[2:], mode='bilinear')
        res8x = torch.add(res16x, res8x)
        res8x = self.dense_4(res8x) + res8x - res16x
        res8x = self.fusion_4(res8x, feature_mem_up)
        feature_mem_up.append(res8x)

        res8x = self.convd8x(res8x)
        res8x = F.upsample(res8x, res4x.size()[2:], mode='bilinear')
        res4x = torch.add(res8x, res4x)
        res4x = self.dense_3(res4x) + res4x - res8x
        res4x = self.fusion_3(res4x, feature_mem_up)
        feature_mem_up.append(res4x)

        res4x = self.convd4x(res4x)
        res4x = F.upsample(res4x, res2x.size()[2:], mode='bilinear')
        res2x = torch.add(res4x, res2x)
        res2x = self.dense_2(res2x) + res2x - res4x
        res2x = self.fusion_2(res2x, feature_mem_up)
        feature_mem_up.append(res2x)

        res2x = self.convd2x(res2x)
        res2x = F.upsample(res2x, x.size()[2:], mode='bilinear')
        x = torch.add(res2x, x)
        x = self.dense_1(x) + x - res2x
        x = self.fusion_1(x, feature_mem_up)

        x = self.conv_output(x)

        return x


class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        out = self.conv2d(x)
        return out


# class ResidualBlock(torch.nn.Module):
#     def __init__(self, channels):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
#         self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
#         self.relu = nn.PReLU()
#
#     def forward(self, x):
#         residual = x
#         out = self.relu(self.conv1(x))
#         out = self.conv2(out) * 0.1
#         out = torch.add(out, residual)
#         return out


# class RCAN(nn.Module):
#     def __init__(self, args):
#         super(RCAN, self).__init__()
#         nChannel = args.nchannel
#         scale = args.scale
#         self.args = args
#
#         # Define Network
#         # ===========================================
#         self.relu = nn.ReLU()
#         self.conv1 = nn.Conv2d(nChannel, 64, kernel_size=7, padding=3)
#         # self.RG1 = residual_group(64, 64)
#         # self.RG2 = residual_group(64, 64)
#         # # self.RG3 = residual_group(64, 64)
#         self.SCAB1 = SCA_block(64, 64)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
#         self.conv4 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
#         # self.reset_params()
#         # ===========================================
#
#     def forward(self, x):
#         # Make a Network path
#         # ===========================================
#         x = self.relu(self.conv1(x))
#
#         sca1 = self.SCAB1(x)
#         sca2 = self.SCAB1(sca1)
#         sca3 = self.SCAB1(sca2)
#         sca3 = sca3 + sca2
#         sca4 = self.SCAB1(sca3)
#         sca4 = sca4 + sca1
#         sca5 = self.SCAB1(sca4)
#         sca5 = sca5 + x
#
#         x = self.relu(self.conv3(sca5))
#         # x = self.pixel_shuffle(x)
#
#         x = self.conv4(x)
#         # ===========================================
#         return x
#
#     # @staticmethod
#     # def weight_init(m):
#     #     if isinstance(m, nn.Conv2d):
#     #         init.xavier_normal_(m.weight)
#     #         # init.constant(m.bias, 0)
#     #
#     # def reset_params(self):
#     #     for i, m in enumerate(self.modules()):
#     #         self.weight_init(m)


class RCAB(nn.Module):
    def __init__(self, channels):
        super(RCAB, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.ca_block = CA_block(channels, channels)
        # self.reset_params()

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.conv2(conv1)
        ca = self.ca_block(conv2)
        return x + ca

    # @staticmethod
    # def weight_init(m):
    #     if isinstance(m, nn.Conv2d):
    #         init.xavier_normal_(m.weight)
    #         # init.constant(m.bias, 0)
    #
    # def reset_params(self):
    #     for i, m in enumerate(self.modules()):
    #         self.weight_init(m)

class CA_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CA_block, self).__init__()
        # global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_down_up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//16, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//16, out_channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_down_up(y)
        return x * y


class conv_layer(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(conv_layer, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation != 'no':
            return self.act(out)
        else:
            return out

class deconv_layer(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):
        super(deconv_layer, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class Decoder_MDCBlock1(torch.nn.Module):
    def __init__(self, num_filter, num_ft, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None, mode='iter1'):
        super(Decoder_MDCBlock1, self).__init__()
        self.mode = mode
        self.num_ft = num_ft - 1
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(self.num_ft):
            self.down_convs.append(
                conv_layer(num_filter * (2 ** i), num_filter * (2 ** (i + 1)), kernel_size, stride, padding, bias, activation, norm=None)
            )
            self.up_convs.append(
                deconv_layer(num_filter * (2 ** (i + 1)), num_filter * (2 ** i), kernel_size, stride, padding, bias, activation, norm=None)
            )

    def forward(self, ft_h, ft_l_list):
        if self.mode == 'iter1' or self.mode == 'conv':
            ft_h_list = []
            for i in range(len(ft_l_list)):
                ft_h_list.append(ft_h)
                ft_h = self.down_convs[self.num_ft- len(ft_l_list) + i](ft_h)

            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft_fusion = self.up_convs[self.num_ft-i-1](ft_fusion - ft_l_list[i]) + ft_h_list[len(ft_l_list)-i-1]

        if self.mode == 'iter2':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_fusion
                for j in range(self.num_ft - i):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[i]
                for j in range(self.num_ft - i):
                    ft = self.up_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter3':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_fusion
                for j in range(i+1):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[len(ft_l_list) - i - 1]
                for j in range(i+1):
                    # print(j)
                    ft = self.up_convs[i + 1 - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter4':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_h
                for j in range(self.num_ft - i):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[i]
                for j in range(self.num_ft - i):
                    ft = self.up_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        return ft_fusion

class Encoder_MDCBlock1(torch.nn.Module):
    def __init__(self, num_filter, num_ft, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None, mode='iter1'):
        super(Encoder_MDCBlock1, self).__init__()
        self.mode = mode
        self.num_ft = num_ft - 1
        self.up_convs = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        for i in range(self.num_ft):
            self.up_convs.append(
                deconv_layer(num_filter // (2 ** i), num_filter // (2 ** (i + 1)), kernel_size, stride, padding, bias, activation, norm=None)
            )
            self.down_convs.append(
                conv_layer(num_filter // (2 ** (i + 1)), num_filter // (2 ** i), kernel_size, stride, padding, bias, activation, norm=None)
            )

    def forward(self, ft_l, ft_h_list):
        if self.mode == 'iter1' or self.mode == 'conv':
            ft_l_list = []
            for i in range(len(ft_h_list)):
                ft_l_list.append(ft_l)
                ft_l = self.up_convs[self.num_ft- len(ft_h_list) + i](ft_l)

            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft_fusion = self.down_convs[self.num_ft-i-1](ft_fusion - ft_h_list[i]) + ft_l_list[len(ft_h_list)-i-1]

        if self.mode == 'iter2':
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_fusion
                for j in range(self.num_ft - i):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[i]
                for j in range(self.num_ft - i):
                    # print(j)
                    ft = self.down_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter3':
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_fusion
                for j in range(i+1):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[len(ft_h_list) - i - 1]
                for j in range(i+1):
                    # print(j)
                    ft = self.down_convs[i + 1 - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter4':
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_l
                for j in range(self.num_ft - i):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[i]
                for j in range(self.num_ft - i):
                    # print(j)
                    ft = self.down_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        return ft_fusion