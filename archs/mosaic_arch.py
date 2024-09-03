import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
import gc

gc.collect()
torch.cuda.empty_cache()


class DenseBlocksTemporalReduce(nn.Module):
    """A concatenation of 3 dense blocks with reduction in temporal dimension.

    Note that the output temporal dimension is 6 fewer the input temporal dimension, since there are 3 blocks.

    Args:
        num_feat (int): Number of channels in the blocks. Default: 64.
        num_grow_ch (int): Growing factor of the dense blocks. Default: 32
        adapt_official_weights (bool): Whether to adapt the weights translated from the official implementation.
            Set to false if you want to train from scratch. Default: False.
    """

    def __init__(self, num_feat=64, num_grow_ch=32, adapt_official_weights=False):
        super(DenseBlocksTemporalReduce, self).__init__()
        if adapt_official_weights:
            eps = 1e-3
            momentum = 1e-3
        else:  # pytorch default values
            eps = 1e-05
            momentum = 0.1

        self.temporal_reduce1 = nn.Sequential(
            nn.BatchNorm3d(num_feat, eps=eps, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_feat, num_feat, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True),
            nn.BatchNorm3d(num_feat, eps=eps, momentum=momentum), nn.ReLU(inplace=True),
            # 时间维度减
            # nn.Conv3d(num_feat, num_grow_ch, (3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True))
            # 时间维度不减
            nn.Conv3d(num_feat, num_grow_ch, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True))

        self.temporal_reduce2 = nn.Sequential(
            nn.BatchNorm3d(num_feat + num_grow_ch, eps=eps, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_feat + num_grow_ch, num_feat + num_grow_ch, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True),
            nn.BatchNorm3d(num_feat + num_grow_ch, eps=eps, momentum=momentum), nn.ReLU(inplace=True),
            # 时间维度减
            # nn.Conv3d(num_feat + num_grow_ch, num_grow_ch, (3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True))
            # 时间维度不减
            nn.Conv3d(num_feat + num_grow_ch, num_grow_ch, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True))

        self.temporal_reduce3 = nn.Sequential(
            nn.BatchNorm3d(num_feat + 2 * num_grow_ch, eps=eps, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_feat + 2 * num_grow_ch, num_feat + 2 * num_grow_ch, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True),
            nn.BatchNorm3d(num_feat + 2 * num_grow_ch, eps=eps, momentum=momentum),
            nn.ReLU(inplace=True),
            # 时间维度减
            # nn.Conv3d(num_feat + 2 * num_grow_ch, num_grow_ch, (3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True))
            # 时间维度不减
            nn.Conv3d(num_feat + 2 * num_grow_ch, num_grow_ch, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True))

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor with shape (b, num_feat, t, h, w).

        Returns:
            t由7变为1——Tensor: Output with shape (b, num_feat + num_grow_ch * 3, t, h, w).
            t不变——Tensor: Output with shape (b, num_feat + num_grow_ch * 3, t, h, w).
        """
        """t由7变为1"""
        # # 输入的x                  [4, 160, 7, 32, 32] 这是dense_block1(x)最后的输出
        # # temporal_reduce1的输出   [4, 32, 5, 32, 32]
        # # cat1的输出               [4, 192, 5, 32, 32] 第一维度：160+32
        # print('输入x', x.shape)
        # x1 = self.temporal_reduce1(x)
        # x1 = torch.cat((x[:, :, 1:-1, :, :], x1), 1)
        # print('x1', x1.shape)
        #
        # # temporal_reduce2的输出   [4, 32, 3, 32, 32]
        # # cat2的输出               [4, 224, 5, 32, 32] 第一维度：192+32
        # x2 = self.temporal_reduce2(x1)
        # x2 = torch.cat((x1[:, :, 1:-1, :, :], x2), 1)
        # print('x2', x2.shape)
        #
        # # temporal_reduce3的输出   [4, 32, 1, 32, 32]
        # # cat3的输出               [4, 256, 1, 32, 32] 第一维度：224+32
        # x3 = self.temporal_reduce3(x2)
        # x3 = torch.cat((x2[:, :, 1:-1, :, :], x3), 1)
        # print('x3', x3.shape)
        """t不变"""
        # 输入的x                  size=16,batch=4:[4, 160, 25, 4, 4] 这是dense_block1(x)最后的输出
        # temporal_reduce1的输出   [4, 32, 25, 8, 8]
        # cat1的输出               [4, 192, 25, 8, 8] 第一维度：160+32
        # print('输入x', x.shape)
        x1 = self.temporal_reduce1(x)
        # print('x1', x1.shape)
        x1 = torch.cat((x, x1), 1)
        # print('x1', x1.shape)

        # temporal_reduce2的输出   [4, 32, 3, 32, 32]
        # cat2的输出               [4, 224, 5, 32, 32] 第一维度：192+32
        x2 = self.temporal_reduce2(x1)
        # print('x2', x2.shape)
        x2 = torch.cat((x1, x2), 1)
        # print('x2', x2.shape)

        # temporal_reduce3的输出   [4, 32, 25, 8, 8]
        # cat3的输出               [4, 256, 25, 8, 8] 第一维度：224+32
        x3 = self.temporal_reduce3(x2)
        # print('x3', x3.shape)
        x3 = torch.cat((x2, x3), 1)
        # print('x3', x3.shape)

        return x3


class DenseBlocks(nn.Module):
    """ A concatenation of N dense blocks.

    Args:
        num_feat (int): Number of channels in the blocks. Default: 64.
        num_grow_ch (int): Growing factor of the dense blocks. Default: 32.
        num_block (int): Number of dense blocks. The values are:
            DUF-S (16 layers): 3
            DUF-M (28 layers): 9
            DUF-L (52 layers): 21
        adapt_official_weights (bool): Whether to adapt the weights translated from the official implementation.
            Set to false if you want to train from scratch. Default: False.
    """

    def __init__(self, num_block, num_feat=64, num_grow_ch=16, adapt_official_weights=False):
        super(DenseBlocks, self).__init__()
        if adapt_official_weights:
            eps = 1e-3
            momentum = 1e-3
        else:  # pytorch default values
            eps = 1e-05
            momentum = 0.1

        self.dense_blocks = nn.ModuleList()
        for i in range(0, num_block):
            self.dense_blocks.append(nn.Sequential(
                nn.BatchNorm3d(num_feat + i * num_grow_ch, eps=eps, momentum=momentum),
                nn.ReLU(inplace=True),
                nn.Conv3d(num_feat + i * num_grow_ch, num_feat + i * num_grow_ch, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True),
                # nn.Conv3d(num_feat + i * num_grow_ch, num_feat + i * num_grow_ch, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True),
                nn.BatchNorm3d(num_feat + i * num_grow_ch, eps=eps, momentum=momentum),
                nn.ReLU(inplace=True),
                # nn.Conv3d(num_feat + i * num_grow_ch, num_grow_ch, (3, 3, 3), stride=(1, 1, 1), padding=(2, 2, 2), bias=True, dilation=(2, 2, 2))
                nn.Conv3d(num_feat + i * num_grow_ch, num_grow_ch, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True)
            ))

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor with shape (b, num_feat, t, h, w).

        Returns:
            Tensor: Output with shape (b, num_feat + num_block * num_grow_ch, t, h, w).
        """
        for i in range(0, len(self.dense_blocks)):
            y = self.dense_blocks[i](x)
            x = torch.cat((x, y), 1)
        return x


class DynamicUpsamplingFilter(nn.Module):
    """Dynamic upsampling filter used in DUF.

    Ref: https://github.com/yhjo09/VSR-DUF.
    It only supports input with 3 channels. And it applies the same filters to 3 channels.

    Args:
        filter_size (tuple): Filter size of generated filters. The shape is (kh, kw). Default: (5, 5).
    """

    def __init__(self, filter_size=(5, 5)):
        super(DynamicUpsamplingFilter, self).__init__()
        if not isinstance(filter_size, tuple):
            raise TypeError(f'The type of filter_size must be tuple, but got type{filter_size}')
        if len(filter_size) != 2:
            raise ValueError(f'The length of filter size must be 2, but got {len(filter_size)}.')

        # generate a local expansion filter, similar to im2col 生成一个本地扩展过滤器，类似于im2col
        # 对于输入数据，根据卷积核大小，将三个通道依次展开为一维数组，形成一个横着的一维长数组；对于卷积核则排成竖着的一维长数组
        # csdn搜索‘【深度学习】基于im2col的展开Python实现卷积层和池化层’有可视化过程

        """滤波器组生成过程"""
        # filter_prod=*filter_size=25, expansion_filter[75, 1, 1, 5, 5], expansion_filter_addtime[75, 1, 25, 5, 5]
        # 首先得到filter_prod=25，np.prod(filter_size)返回指定轴上元素的乘积，不指定轴默认是所有元素的乘积,此处返回所有元素乘积25
        # 再通过torch.eye(int(filter_prod))生成25*25的对角矩阵y，再reshape变为[25, 1, 5, 5]数组expansion_filter，并在第0维重复三次输出[75, 1, 5, 5]
        # 在expansion_filter的第2维增加时间维度，得到expansion_filter_addtime[75, 1, 1, 5, 5]，最后第2维重复25次[75, 1, 25, 5, 5]
        # 其中：*filter_size，表示5*5的二维数组按照位置相乘，每一个5*5的数组中只有1个元素1，其余为0，
        #      也就是第一个5*5矩阵00位置为1，其余0；第二个矩阵01位置为1，其余0；...；第25个矩阵55位置为1，其余0
        # 重复三次操作相当于对三个通道都进行滤波，输出[75, 1, 1, 5, 5]；重复25次相当于对25个输入都滤波，输出[75, 1, 25, 5, 5]
        # 说明：最后输出的滤波器组有75个[5, 5]的二维数组排成一列，每一个[5, 5]的二维数组按照位置顺序保一个元素1，其余为0，如上面’其中‘的说明
        #      形状同20221014报告中的2.4.2的im2col的卷积核的操作

        self.filter_size = filter_size
        filter_prod = np.prod(filter_size)
        y = torch.eye(int(filter_prod))
        expansion_filter = y.view(filter_prod, 1, *filter_size)
        expansion_filter = expansion_filter.repeat(3, 1, 1, 1)
        expansion_filter_addtime = torch.unsqueeze(expansion_filter, dim=2)
        # print('expansion_filter_addtime1', expansion_filter_addtime.shape)
        self.expansion_filter = expansion_filter_addtime.repeat(1, 1, 25, 1, 1)
        # print('滤波组expansion_filter', self.expansion_filter.shape)

    def forward(self, x, filters):
        """Forward function for DynamicUpsamplingFilter.

        Args:
            t由7变为1——x (Tensor): Input image with 3 channels. The shape is (n, 3, h, w).
            t不变——x (Tensor): Input image with 3 channels. The shape is (n, 3, t, h, w).

            filters (Tensor): Generated dynamic filters.
                t由7变为1——The shape is (n, filter_prod, upsampling_square, h, w).
                t不变——The shape is (n, filter_prod, upsampling_square, t, h, w).

                filter_prod: prod of filter kernel size, e.g., 1*5*5=25.
                upsampling_square: similar to pixel shuffle, upsampling_square = upsampling * upsampling
                    e.g., for x 4 upsampling, upsampling_square= 4*4 = 16

        Returns:
            t由7变为1——Tensor: Filtered image with shape (n, 3*upsampling_square, h, w)
            t不变——Tensor: Filtered image with shape (n, 3*upsampling_square, t, h, w)
        """
        n, filter_prod, upsampling_square, t, h, w = filters.size()
        kh, kw = self.filter_size
        expanded_input = torch.zeros(n, 3*filter_prod, t, h, w)

        # 以下upsampling_square=1
        # 输入x[4, 3, 25, 8, 8], filters[4, 25, 16, 8, 8], expansion_filter滤波输出[75, 1, 25, 5, 5],
        # x滤波后输出y1 = self.expansion_filter.to(x)，y1[75, 1, 25, 5, 5]
        # 分组卷积（GPU并行计算，提速度）时有两个方式：
        # 1.25个时间维度卷积，groups=3时，输入x与y1卷积输出expanded_input[4, 75, 5, 8, 8]，其中groups=2、4、5都不能输出
        # 2.单个时间维度来处理卷积，重复25次，得到输出[4, 75, 25, 8, 8]
        # expanded_input通过view()reshape为[4, 3, 25, 25, 8, 8]，再重新排列为[4, 25, 8, 8, 3, 25] 75=3*25=3*5*5
        # filters由原来的[4, 25, 16, 25, 8, 8]再次reshape为[4, 25, 8, 8, 25, 16]
        # out:expanded_input[4, 25, 8, 8, 3, 25]和filters[4, 25, 8, 8, 25, 16]通过高维数组相乘函数matmul得到,输出[4, 25, 8, 8, 3, 16]
        # out再reshape为[4, 3, 16, 25, 8, 8],再合并通道最后输出[4, 3*16, 25, 8, 8]
        # 相乘过程：前三维对应相乘，最后两个维度(3, filter_prod)×(filter_prod, upsampling_square)=(3, upsampling_square)

        y1 = self.expansion_filter.to(x)
        for i in range(0, 25):
            expanded_input[:, :, i, :, :] = F.conv2d(x[:, :, i, :, :], y1[:, :, i, :, :], padding=(kh // 2, kw // 2), groups=3)
        # expanded_input = F.conv3d(x, y1, padding=(kh // 2, kw // 2, kh // 2), groups=3)
        # print('x与y1卷积后', expanded_input.shape)
        expanded_input = expanded_input.view(n, 3, filter_prod, 25, h, w).permute(0, 3, 4, 5, 1, 2).cuda()
        filters = filters.permute(0, 3, 4, 5, 1, 2)
        # print('filters', filters.shape)
        out = torch.matmul(expanded_input, filters)

        return out.permute(0, 4, 5, 1, 2, 3).view(n, 3 * upsampling_square, 25, h, w)


@ARCH_REGISTRY.register()
class DUFMOSAIC(nn.Module):
    """Network architecture for DUF

    Paper: Jo et.al. Deep Video Super-Resolution Network Using Dynamic
            Upsampling Filters Without Explicit Motion Compensation, CVPR, 2018
    Code reference:
        https://github.com/yhjo09/VSR-DUF
    For all the models below, 'adapt_official_weights' is only necessary when
    loading the weights converted from the official TensorFlow weights.
    Please set it to False if you are training the model from scratch.

    There are three models with different model size: DUF16Layers, DUF28Layers,
    and DUF52Layers. This class is the base class for these models.

    Args:
        scale (int): The upsampling factor. Default: 4.
        num_layer (int): The number of layers. Default: 52.
        adapt_official_weights_weights (bool): Whether to adapt the weights
            translated from the official implementation. Set to false if you
            want to train from scratch. Default: False.
    """

    def __init__(self, scale=4, num_layer=52, adapt_official_weights=False):
        super(DUFMOSAIC, self).__init__()
        self.scale = scale
        if adapt_official_weights:
            eps = 1e-3
            momentum = 1e-3
        else:  # pytorch default values
            eps = 1e-05
            momentum = 0.1

        # 原始
        # self.conv3d1 = nn.Conv3d(3, 64, (3, 3, 3), stride=(1, 1, 1), padding=(2, 2, 2), bias=True, dilation=(2, 2, 2))
        self.conv3d1 = nn.Conv3d(3, 64, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True)

        # dynamic_filter(n, filter_prod, upsampling_square, h, w)
        self.dynamic_filter = DynamicUpsamplingFilter((5, 5))

        if num_layer == 16:
            num_block = 3
            num_grow_ch = 32
        elif num_layer == 28:
            num_block = 9
            num_grow_ch = 16
        elif num_layer == 52:
            num_block = 21
            num_grow_ch = 16
        else:
            raise ValueError(f'Only supported (16, 28, 52) layers, but got {num_layer}.')
        # dense_block1(b, num_feat + num_block * num_grow_ch, t, h, w)
        self.dense_block1 = DenseBlocks(
            num_block=num_block, num_feat=64, num_grow_ch=num_grow_ch, adapt_official_weights=adapt_official_weights)  # T = 7

        # dense_block2(b, num_feat + num_grow_ch * 3, 1, h, w)
        self.dense_block2 = DenseBlocksTemporalReduce(
            64 + num_grow_ch * num_block, num_grow_ch, adapt_official_weights=adapt_official_weights)  # T = 1
        channels = 64 + num_grow_ch * num_block + num_grow_ch * 3

        self.bn3d2 = nn.BatchNorm3d(channels, eps=eps, momentum=momentum)
        self.conv3d2 = nn.Conv3d(channels, 256, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True)

        self.conv3d_r1 = nn.Conv3d(256, 256, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        if self.scale == 1:
            self.conv3d_r2 = nn.Conv3d(256, 3 * (scale), (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        else:
            self.conv3d_r2 = nn.Conv3d(256, 3 * (scale**2), (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)

        self.conv3d_f1 = nn.Conv3d(256, 512, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        if self.scale == 1:
            self.conv3d_f2 = nn.Conv3d(512, 1 * 5 * 5 * (scale), (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        else:
            self.conv3d_f2 = nn.Conv3d(512, 1 * 5 * 5 * (scale**2), (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input with shape (b, 7, c, h, w)

        Returns:
            Tensor: Output with shape (b, c, h * scale, w * scale) 有放大倍数
            Tensor: Output with shape (b, c, h * scale, w * scale) 没有放大倍数
        """

        def featuremap_visual(feature,
                              out_dir=None,  # 特征图保存路径文件
                              save_feature=True,  # 是否以图片形式保存特征图
                              show_feature=True,  # 是否使用plt显示特征图
                              feature_title=None,  # 特征图名字，默认以shape作为title
                              num_ch=-1,  # 显示特征图前几个通道，-1 or None 都显示
                              nrow=2,  # 每行显示多少个特征图通道
                              padding=10,  # 特征图之间间隔多少像素值
                              pad_value=1  # 特征图之间的间隔像素
                              ):
            import matplotlib.pylab as plt
            import torchvision
            import os
            # feature = feature.detach().cpu()
            b, c, h, w = feature.shape
            feature = feature[0]
            feature = feature.unsqueeze(1)

            if c > num_ch > 0:
                feature = feature[:num_ch]

            img = torchvision.utils.make_grid(feature, nrow=nrow, padding=padding, pad_value=pad_value)
            img = img.detach().cpu()
            img = img.numpy()
            images = img.transpose((1, 2, 0))

            # title = str(images.shape) if feature_title is None else str(feature_title)
            title = str('hwc-') + str(h) + '-' + str(w) + '-' + str(c) if feature_title is None else str(feature_title)

            plt.title(title)
            plt.imshow(images)
            if save_feature:
                # root=r'C:\Users\Administrator\Desktop\CODE_TJ\123'
                # plt.savefig(os.path.join(root,'1.jpg'))
                out_root = title + '.jpg' if out_dir == '' or out_dir is None else os.path.join(out_dir, title + '.jpg')
                plt.savefig(out_root)

            if show_feature:        plt.show()

        num_batches, num_imgs, _, h, w = x.size()
        outs = []

        """t由7变为1"""
        # x = x.permute(0, 2, 1, 3, 4)  # (b, c, 7, h, w) for Conv3D [4, 3, 7, 32, 32]
        # print('输入x', x.shape)
        # x_center = x[:, :, num_imgs // 2, :, :]  # [4, 3, 32, 32]
        #
        # x = self.conv3d1(x)  # [4, 64, 7, 32, 32]
        # print('conv3d1', x.shape)
        # x = self.dense_block1(x)  # [4, 160, 7, 32, 32]
        # x = self.dense_block2(x)  # [4, 256, 1, 32, 32]
        # print('dense_block2', x.shape)
        # x = F.relu(self.bn3d2(x), inplace=True)
        # x = F.relu(self.conv3d2(x), inplace=True)  # [4, 256, 1, 32, 32]
        # print('conv3d2', x.shape)
        #
        # # residual image
        # res = self.conv3d_r2(F.relu(self.conv3d_r1(x), inplace=True))  # [4, 3, 1, 32, 32]
        # print('res', res.shape)
        #
        # # filter 得到的滤波器组filter_ [4, 25, 1, 32, 32]
        # filter_ = self.conv3d_f2(F.relu(self.conv3d_f1(x), inplace=True))
        # print('filter_', filter_.shape)
        #
        # if self.scale == 1:
        #     filter_ = F.softmax(filter_.view(num_batches, 25, self.scale, h, w), dim=1)
        # else:
        #     filter_ = F.softmax(filter_.view(num_batches, 25, self.scale**2, h, w), dim=1)
        # print('sofmax后filter_', filter_.shape)
        #
        # # dynamic filter
        # out = self.dynamic_filter(x_center, filter_)  # [4, 3, 32, 32]
        # print('y1', out.shape)
        # # squeeze_()函数：将输入张量形状中的1去除并返回，例1。 当给定dim时，那么挤压操作只在给定维度上，例2。
        # # 例1：输入(A×1×B×1×C×1×D)，那么输出为：(A×B×C×D)。
        # # 例2：输入(A×1×B), squeeze(input, 0)，输出将会保持张量不变；squeeze(input, 1)，输出变成(A×B)。
        # out += res.squeeze_(2)  # res去掉第2维的维度之后加上滤波后的结果
        # print('squeeze后的out', out.shape)
        # out = F.pixel_shuffle(out, self.scale)  # [4, 3, 32, 32]
        # print('out', out.shape)

        """t不变"""
        x = x.permute(0, 2, 1, 3, 4)
        # 输入x(b, c, 7, h=gt_size/scale, w=gt_size/scale) for Conv3D gt_size16[4, 3, 25, 4, 4]
        # print('输入x', x.shape)
        x_25input = x
        final_out = torch.zeros(num_batches, num_imgs, 3, h*self.scale, h*self.scale)

        x = self.conv3d1(x)  # size16batch=4:[4, 64, 25, 4, 4]  batch=1:[1, 64, 25, 4, 4]
        # print('conv3d1', x.shape)
        x = self.dense_block1(x)  # batch=4:[4, 160, 25, 4, 4] batch=1:[1, 160, 25, 4, 4]
        # print('dense_block1', x.shape)
        x = self.dense_block2(x)  # [4, 256, 25, 8, 8] 16:[4, 256, 25, 4, 4]  batch=1:[1, 256, 25, 2, 2]
        # print('dense_block2', x.shape)
        x = F.relu(self.bn3d2(x), inplace=True)
        # print('bn3d2', x.shape)
        x = F.relu(self.conv3d2(x), inplace=True)  # [4, 256, 25, 8, 8]

        # residual image  [4, 48, 25, 8, 8]
        x = F.relu(self.conv3d_r1(x), inplace=True)  # [4, 256, 25, 8, 8] 16:[4, 256, 25, 4, 4]
        # print('conv3d_r1', x.shape)
        res = self.conv3d_r2(x)  # [4, 48, 25, 8, 8]  48=3*(4*4)  16:[4, 48, 25, 4, 4]
        outs.append(res)
        featuremap_visual(res)
        # print('res', res.shape)

        # filter 得到的滤波器组filter_ [4, 400, 25, 8, 8] conv3d_f1的输出[4, 512, 25, 8, 8]
        filter_ = self.conv3d_f2(F.relu(self.conv3d_f1(x), inplace=True))  # [4, 400, 25, 8, 8] 400=5*5*(4*4) 16:[4, 400, 25, 4, 4]
        # print('filter_', filter_.shape)  验证时[1, 400, 25, 128, 128]

        # if self.scale == 1:
        #     filter_ = F.softmax(filter_.view(num_batches, 25, self.scale, h, w), dim=1)
        # else:
        #     filter_ = F.softmax(filter_.view(num_batches, 25, self.scale ** 2, h, w), dim=1)
        # 原本filter_的输出是[4, 25, 1, 32, 32]，即第2维是时间维，维度为1，即只有1帧输出，
        # 但现在是25输出，并且filter_输出是[4, 400, 25, 8, 8]，所以在原来的基础上多加了一个时间维
        # 最后filter_重新整形为[4, 25, 16, 25, 8, 8]，才进行下一步的softmax归一化
        if self.scale == 1:
            filter_ = F.softmax(filter_.view(num_batches, 25, self.scale, 25, h, w), dim=1)
        else:
            filter_ = F.softmax(filter_.view(num_batches, 25, self.scale ** 2, 25, h, w), dim=1)
        outs.append(filter_)
        featuremap_visual(filter_)
        # print('sofmax后filter_', filter_.shape)

        # dynamic filter
        # 输入x_25input[4, 3, 25, 8, 8]，filter_ 32[4, 25, 16, 25, 8, 8] 16[4, 25, 16, 25, 4, 4]，out[4, 48, 25, 8, 8]，res[4, 48, 25, 8, 8]
        out = self.dynamic_filter(x_25input, filter_)  # 16:[4, 48, 25, 4, 4]
        # print('y1', out.shape)
        out += res  # [4, 48, 25, 8, 8]  16:[4, 48, 25, 4, 4]
        # print('没有squeeze的res+out', out.shape)
        # 原代码
        # out = self.dynamic_filter(x, filter_)  # x[4, 3, 32, 32]
        # out += res.squeeze_(2)  # res去掉第2维的维度之后加上滤波后的结果
        # out = F.pixel_shuffle(out, self.scale)  # [4, 3, 32, 32]
        # squeeze_()函数：将输入张量形状中的1去除并返回，例1。 当给定dim时，那么挤压操作只在给定维度上，例2。
        # 例1：输入(A×1×B×1×C×1×D)，那么输出为：(A×B×C×D)。
        # 例2：输入(A×1×B), squeeze(input, 0)，输出将会保持张量不变；squeeze(input, 1)，输出变成(A×B)。
        for i in range(0, 25):
            final_out[:, i, :, :, :] = F.pixel_shuffle(out[:, :, i, :, :], self.scale)  # [4, 25, 3, 8, 8]  测试输出[1, 48, 25, 128, 128]
        # print('out', out.shape)
        # GPU模式下
        return final_out.cuda()
        # return final_out




