

import torch.nn as nn
import torch
from collections import OrderedDict
from torch.nn import functional as F


def Conv(in_planes, out_planes, **kwargs):
    "3x3 convolution with padding"
    padding = kwargs.get('padding', 1)
    bias = kwargs.get('bias', False)
    stride = kwargs.get('stride', 1)
    kernel_size = kwargs.get('kernel_size', 3)
    out = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    return out


class DenseBlocksTemporalReduce(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(DenseBlocksTemporalReduce, self).__init__()
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

    def __init__(self, num_block, num_feat=64, num_grow_ch=16):
        super(DenseBlocks, self).__init__()
        eps = 1e-05
        momentum = 0.1

        self.dense_blocks = nn.ModuleList()
        for i in range(0, num_block):
            self.dense_blocks.append(nn.Sequential(
                nn.BatchNorm3d(num_feat + i * num_grow_ch, eps=eps, momentum=momentum),
                nn.ReLU(inplace=True),
                nn.Conv3d(num_feat + i * num_grow_ch, num_feat + i * num_grow_ch, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True),
                nn.BatchNorm3d(num_feat + i * num_grow_ch, eps=eps, momentum=momentum),
                nn.ReLU(inplace=True),
                nn.Conv3d(num_feat + i * num_grow_ch, num_grow_ch, (3, 3, 3), stride=(1, 1, 1), padding=(2, 2, 2), bias=True, dilation=(2, 2, 2))
                # nn.Conv3d(num_feat + i * num_grow_ch, num_grow_ch, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True)
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


class dufnet(nn.Module):
    def __init__(self,
                 scale=4,
                 num_layer=52,
                 eps=1e-05,
                 momentum=0.1,
                 pretrained=None,
                 frozen_stages=-1
                 # num_classes=None
                 ):
        super(dufnet, self).__init__()
        self.scale = scale
        self.num_layer = num_layer
        self.eps = eps
        self.momentum = momentum
        self.frozen_stages = frozen_stages
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
        self.dense_block1 = DenseBlocks(num_block=num_block, num_feat=64, num_grow_ch=num_grow_ch)  # T = 7

        # dense_block2(b, num_feat + num_grow_ch * 3, 1, h, w)
        self.dense_block2 = DenseBlocksTemporalReduce(64 + num_grow_ch * num_block, num_grow_ch)  # T = 1
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
        self._freeze_stages()  # 冻结函数
        if pretrained is not None:
            self.init_weights(pretrained=pretrained)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            self.load_checkpoint(pretrained)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
                    if hasattr(m, 'bias') and m.bias is not None:  # m包含该属性且m.bias非None # hasattr(对象，属性)表示对象是否包含该属性
                        nn.init.constant_(m.bias, 0)

                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def load_checkpoint(self, pretrained):

        checkpoint = torch.load(pretrained)
        if isinstance(checkpoint, OrderedDict):
            state_dict = checkpoint
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']

        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}

        unexpected_keys = []  # 保存checkpoint不在module中的key
        model_state = self.state_dict()  # 模型变量

        for name, param in state_dict.items():  # 循环遍历pretrained的权重
            if name not in model_state:
                unexpected_keys.append(name)
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data

            try:
                model_state[name].copy_(param)  # 试图赋值给模型
            except Exception:
                raise RuntimeError(
                    'While copying the parameter named {}, '
                    'whose dimensions in the model are {} not equal '
                    'whose dimensions in the checkpoint are {}.'.format(
                        name, model_state[name].size(), param.size()))
        missing_keys = set(model_state.keys()) - set(state_dict.keys())
        print('missing_keys:', missing_keys)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        outs = []
        num_batches, num_imgs, _, h, w = x.size()

        # x = x.permute(0, 2, 1, 3, 4)
        x_25input = x
        final_out = torch.zeros(num_batches, num_imgs, 3, h * self.scale, h * self.scale)

        x = self.conv3d1(x)  # size16batch=4:[4, 64, 25, 4, 4]  batch=1:[1, 64, 25, 4, 4]
        outs.append(x)
        featuremap_visual(x)
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
        # print('res', res.shape)
        outs.append(x)
        featuremap_visual(x)
        # filter 得到的滤波器组filter_ [4, 400, 25, 8, 8] conv3d_f1的输出[4, 512, 25, 8, 8]
        filter_ = self.conv3d_f2(
            F.relu(self.conv3d_f1(x), inplace=True))  # [4, 400, 25, 8, 8] 400=5*5*(4*4) 16:[4, 400, 25, 4, 4]
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
        # print('sofmax后filter_', filter_.shape)
        outs.append(filter_)
        featuremap_visual(filter_)
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
            final_out[:, i, :, :, :] = F.pixel_shuffle(out[:, :, i, :, :],
                                                       self.scale)  # [4, 25, 3, 8, 8]  测试输出[1, 48, 25, 128, 128]
        # print('out', out.shape)
        # GPU模式下
        return tuple(outs)


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


import cv2
import numpy as np


def imnormalize(img,
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True
                ):
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return (img - mean) / std


if __name__ == '__main__':
    import matplotlib.pylab as plt
    import os
    import numpy as np

    path = r'D:\BasicSR\datasets\test256\LR_256\000'
    imglist = os.listdir(path)
    imglist.sort(key=lambda x: int(x.split('.')[0]))  # 按照数字进行排序后按顺序读取文件夹下的图片
    # imgs = []
    imgs = np.zeros((1, 256, 256, 25), dtype=np.float32)
    # print(imglist)
    for i in range(25):
        imgpath = os.path.join(path, imglist[i])
        # print(imgpath)
        img = cv2.imread(imgpath)  # 读取图片
        img = imnormalize(img)
        imgs[0, :, :, i] = img[:,:,0]
    imgs = torch.from_numpy(imgs)
    imgs = torch.unsqueeze(imgs, 0)
    imgs = imgs.permute(0, 2, 1, 3, 4)
    imgs = torch.tensor(imgs, dtype=torch.float32)
    imgs = imgs.to('cuda:0')
    # imgs.append(img)
    model = dufnet(scale=2, num_layer=28)
    # model.init_weights(pretrained='./resnet50.pth')  # 可以使用，也可以注释
    model = model.cuda()
    out = model(imgs)

    # 单张图片
    # img = cv2.imread(r'D:\BasicSR\datasets\test256\LR_256\000\00000001.png')  # 读取图片
    # img = imnormalize(img)
    # img = torch.from_numpy(img)
    #
    # img = torch.unsqueeze(img, 0)
    # img = img.permute(0, 3, 1, 2)
    # img = torch.tensor(img, dtype=torch.float32)
    # img = img.to('cuda:0')
    #
    # model = dufnet(scale=2, num_layer=28)
    # # model.init_weights(pretrained='./resnet50.pth')  # 可以使用，也可以注释
    # model = model.cuda()
    # out = model(img)