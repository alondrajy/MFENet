import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *





###################################################################################加小波变换

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2  # 4,3,128,256
    x02 = x[:, :, 1::2, :] / 2  # 4,3,128,256
    x1 = x01[:, :, :, 0::2]  # 4,3,128,128
    x2 = x02[:, :, :, 0::2]  # 4,3,128,128
    x3 = x01[:, :, :, 1::2]  # 4,3,128,128
    x4 = x02[:, :, :, 1::2]  # 4,3,128,128
    x_LL = x1 + x2 + x3 + x4  # 4,3,128,128
    x_HL = -x1 - x2 + x3 + x4  # 4,3,128,128
    x_LH = -x1 + x2 - x3 + x4  # 4,3,128,128
    x_HH = x1 - x2 - x3 + x4  # 4,3,128,128

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)

# 使用哈尔 haar 小波变换来实现二维离散小波
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = int(in_batch / (r ** 2)), in_channel, r * in_height, r * in_width
    x1 = x[0:out_batch, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

# 二维离散小波
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)

# 逆向二维离散小波
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class BA_Block(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(BA_Block, self).__init__()
        midplanes = int(outplanes//2)


        # self.input_conv = nn.Conv2d(3, inplanes, kernel_size=3, padding=1)
        # self.input_relu = nn.LeakyReLU(0.2, True)

        self.pool_1_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_1_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_1_h = nn.Conv2d(inplanes, midplanes*2, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.conv_1_w = nn.Conv2d(inplanes, midplanes*2, kernel_size=(1, 3), padding=(0, 1), bias=False)

        self.pool_3_h = nn.AdaptiveAvgPool2d((None, 3))
        self.pool_3_w = nn.AdaptiveAvgPool2d((3, None))
        self.conv_3 = nn.Conv2d(inplanes, midplanes*2, kernel_size=3, padding=1, bias=False)

        self.pool_5_h = nn.AdaptiveAvgPool2d((None, 5))
        self.pool_5_w = nn.AdaptiveAvgPool2d((5, None))
        self.conv_5 = nn.Conv2d(inplanes, midplanes*2, kernel_size=3, padding=1, bias=False)

        self.pool_7_h = nn.AdaptiveAvgPool2d((None, 7))
        self.pool_7_w = nn.AdaptiveAvgPool2d((7, None))
        self.conv_7 = nn.Conv2d(inplanes, midplanes*2, kernel_size=3, padding=1, bias=False)

        self.fuse_conv = nn.Conv2d(midplanes * 8, midplanes*2, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=False)
        self.conv_final = nn.Conv2d(midplanes*2, outplanes, kernel_size=1, bias=True)


        self.mask_conv_1 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1)
        self.mask_relu = nn.ReLU(inplace=False)#可以替换成GeLU试试
        self.mask_conv_2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1)
        self.norm ='backward'

        self.main = nn.Sequential(
            BasicConv(inplanes, outplanes, kernel_size=3, stride=1, relu=True),
            BasicConv(outplanes, outplanes, kernel_size=3, stride=1, relu=False)
        )

        self.DWT = DWT()
        self.IWT = IWT()

    def forward(self, x):

        _, _, h, w = x.shape

        x_1_h = self.pool_1_h(x)
        x_1_h = self.conv_1_h(x_1_h)
        x_1_h = x_1_h.expand(-1, -1, h, w)


        x_1_w = self.pool_1_w(x)
        x_1_w = self.conv_1_w(x_1_w)
        x_1_w = x_1_w.expand(-1, -1, h, w)


        x_3_h = self.pool_3_h(x)
        x_3_h = self.conv_3(x_3_h)
        x_3_h = F.interpolate(x_3_h, (h, w))

        x_3_w = self.pool_3_w(x)
        x_3_w = self.conv_3(x_3_w)
        x_3_w = F.interpolate(x_3_w, (h, w))

        x_5_h = self.pool_5_h(x)
        x_5_h = self.conv_5(x_5_h)
        x_5_h = F.interpolate(x_5_h, (h, w))

        x_5_w = self.pool_5_w(x)
        x_5_w = self.conv_5(x_5_w)
        x_5_w = F.interpolate(x_5_w, (h, w))

        x_7_h = self.pool_7_h(x)
        x_7_h = self.conv_7(x_7_h)
        x_7_h = F.interpolate(x_7_h, (h, w))

        x_7_w = self.pool_7_w(x)
        x_7_w = self.conv_7(x_7_w)
        x_7_w = F.interpolate(x_7_w, (h, w))

        y = self.relu(self.fuse_conv(torch.cat((x_1_h + x_1_w, x_3_h + x_3_w, x_5_h + x_5_w, x_7_h + x_7_w),dim=1)))
        y = self.conv_final(y)

        hx = self.DWT(x) 
        hx = self.main(hx)
        hx = self.IWT(hx)  

        hx=hx+y
        hx = self.mask_relu(self.mask_conv_1(hx))
        mask = self.mask_conv_2(hx).sigmoid()
        hx = x* mask

        return hx






class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)




class MT(nn.Module):
    def __init__(self, out_channel, stride=1, kernels=[1, 3, 5, 7]):
        super(MT, self).__init__()

        self.input_conv = nn.Conv2d(3, out_channel, kernel_size=3, padding=1)
        self.input_relu = nn.LeakyReLU(0.2, True)
        padding_dict = {1: 0, 3: 1, 5: 2, 7: 3}
        self.seps = nn.ModuleList()
        for kernel in kernels:
            sep = nn.Conv2d(out_channel, out_channel, kernel_size=kernel, stride=1, padding=padding_dict[kernel],
                            dilation=1, groups=out_channel, bias=False)
            self.seps.append(sep)
        self.bn1 = nn.BatchNorm2d(out_channel * len(kernels))
        self.pointwise = nn.Conv2d(out_channel * len(kernels), out_channel-3, 1, stride, 0, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channel-3)
        self.conv = BasicConv(out_channel, out_channel, kernel_size=1, stride=1, relu=False)
    def forward(self, x):
        hx = self.input_relu((self.input_conv(x)))
        seps = []
        for sep in self.seps:
            seps.append(sep(hx))
        out_seq = torch.cat(seps, dim=1)
        out = self.pointwise(self.bn1(out_seq))
        out = self.bn2(out)
        x = torch.cat([x, out], dim=1)
        x=self.conv(x)
        return x



class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out





############################################################################################################原始模型################################
class MIMOUNet_y(nn.Module):##原始模型
    def __init__(self, num_res=8):
        super(MIMOUNet_y, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)



    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)

        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)


        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)


        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs
#################################################################################################################加BA模型#############################################
class MIMOUNet_BA(nn.Module):#加了BA的模型
    def __init__(self, num_res=8):
        super(MIMOUNet_BA, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)


        self.BA1 = BA_Block(base_channel * 1, base_channel * 1)  # 将BA放在AFF后面
        self.BA2 = BA_Block(base_channel * 2, base_channel * 2)  # 将BA放在AFF后面



    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)

        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        res2 = self.BA2(res2)#64通道
        res1 = self.BA1(res1)#32通道

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs


class MIMOUNet_MT(nn.Module):#SCM变成深度可分离卷积的模型
    def __init__(self, num_res=8):
        super(MIMOUNet_MT, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(base_channel, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])



        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.MT1 = MT(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.MT2 = MT(base_channel * 2)

        self.MT3 = MT(base_channel )

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)

        z2 = self.MT2(x_2)
        z4 = self.MT1(x_4)


        outputs = list()

        x_ = self.MT3(x)
        x_ = self.feat_extract[0](x_)
        res1 = self.Encoder[0](x_)


        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)



        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs


##################################################################################################################两个模块联合模型##########################################################


class MIMOUNet_MT_BA(nn.Module):#两个模块联合
    def __init__(self, num_res=8):
        super(MIMOUNet_MT_BA, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(base_channel, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])
        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.MT1 = MT(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.MT2 = MT(base_channel * 2)
        self.MT3 = MT(base_channel )

        self.BA1 = BA_Block(base_channel * 1, base_channel * 1)  # 将BA放在AFF后面
        self.BA2 = BA_Block(base_channel * 2, base_channel * 2)  # 将BA放在AFF后面



    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)

        z2 = self.MT2(x_2)
        z4 = self.MT1(x_4)

   

        outputs = list()

 
        x_ = self.MT3(x)
        x_ = self.feat_extract[0](x_)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        res2 = self.BA2(res2)#64通道
        res1 = self.BA1(res1)#32通道

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs
###########################################################################################################################PLUS模型#######################################
class MFENet(nn.Module):
    def __init__(self, num_res = 20):
        super(MFENet, self).__init__()
        base_channel = 32
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

         # self.feat_extract = nn.ModuleList([
        #     BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
        #     BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
        #     BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
        #     BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
        #     BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
        #     BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        # ])
        self.feat_extract = nn.ModuleList([
            BasicConv(base_channel, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])


        self.FAM1 = FAM(base_channel * 4)
        self.MT1 = MT(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.MT2 = MT(base_channel * 2)
        self.MT3 = MT(base_channel )

        self.BA1 = BA_Block(base_channel * 1, base_channel * 1)  # 将BA放在AFF后面
        self.BA2 = BA_Block(base_channel * 2, base_channel * 2)  # 将BA放在AFF后面


        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.1)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
    
        z2 = self.MT2(x_2)
        z4 = self.MT1(x_4)

        outputs = list()

        x_ = self.MT3(x)
        x_ = self.feat_extract[0](x_)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        res2 = self.BA2(res2)#64通道
        res1 = self.BA1(res1)#32通道


        res2 = self.drop2(res2)
        res1 = self.drop1(res1)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs


def build_net(model_name):
    class ModelError(Exception):
        def __init__(self, msg):
            self.msg = msg

        def __str__(self):
            return self.msg

    if model_name == "MFENet":
        return MFENet()
    elif model_name == "MIMO-UNet":
        return MIMOUNet_y()
    elif model_name == "MIMO-UNet_BA":
        return MIMOUNet_BA()
    elif model_name == "MIMO-UNet_MT":
        return MIMOUNet_MT()
    elif model_name == "MIMO-UNet_MT_BA":
        return MIMOUNet_MT_BA()
    raise ModelError('Wrong Model!\nYou should choose MIMO-UNetPlus or MIMO-UNet.')
