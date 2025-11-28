from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from numpy.ma.core import shape

from TPPI.models.utils import *
import numpy as np
from torch.nn import functional as F
from einops import rearrange
import auxil
import yaml
import pandas as pd
import warnings
import joblib
import os
import math
import torch
import torch.nn as nn
import torchvision.transforms as T
warnings.filterwarnings(action='ignore')

# 引用注意力机制
from fightingcv_attention.attention.CBAM import CBAMBlock
from fightingcv_attention.attention.SEAttention import SEAttention
from fightingcv_attention.attention.BAM import BAMBlock
from fightingcv_attention.attention.ECAAttention import ECAAttention

# 初始化参数
device = auxil.get_device()

# 常用工具函数
# 字符串转整型
def string_to_int_list(string):
    # 使用逗号分隔字符串
    str_list = string.split(',')
    # 初始化整数列表
    int_list = []

    # 遍历字符串列表
    for item in str_list:
        # 如果字符串表示true，则转换为1
        if item.lower() == "true":
            int_list.append(1)
        # 如果字符串表示false，则转换为0
        elif item.lower() == "false":
            int_list.append(0)
        # 否则，假设字符串是数字，直接转换为整数
        else:
            int_list.append(int(item))
    return int_list

# 引入参数
with open("configs/config.yml") as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)
    modes_number = string_to_int_list(cfg['Data']['modes_number'])
    started = cfg["Data"]["band_selection"][0]
    end = cfg["Data"]["band_selection"][1]
    in_channel = end - started
    cumulative_modes = np.cumsum(modes_number)
    out_class = cfg['Data']['class']
    PPsize = cfg['Preprocessing']['PP_size']

# 神经网络复用模块
    # TODO 测试模块
class PatchExtract(nn.Module):
    def __init__(self, patch_size):
        super(PatchExtract, self).__init__()
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[1]

    def forward(self, images):
        patches = F.pad(images, pad=(0, 1, 0, 1))
        patches = F.unfold(
            patches,
            kernel_size=(self.patch_size_x, self.patch_size_y),
            stride=(self.patch_size_x, self.patch_size_y)
        )
        return patches

class PatchEmbedding(nn.Module):
    def __init__(self, DEM_patch_size,num_patch, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.num_patch = num_patch
        self.proj = nn.Linear(in_features=DEM_patch_size*DEM_patch_size, out_features=embed_dim)  # 注意：in_features 需指定！
        self.pos_embed = nn.Embedding(num_embeddings=self.num_patch, embedding_dim=embed_dim)

    def forward(self, patch):
        """
        patch: Tensor of shape [B, num_patch, patch_dim]
        returns: [B, num_patch, embed_dim]
        """
        B, N, D = patch.shape  # B=batch size, N=num_patch, D=patch_dim
        pos = torch.arange(self.num_patch, device=patch.device)  # [0, 1, ..., num_patch-1]
        pos = self.pos_embed(pos)  # [num_patch, embed_dim]
        x = self.proj(patch)  # [B, num_patch, embed_dim]
        return x + pos  # broadcasting

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W, C):
    B = int(windows.shape[0])
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.):
        super(WindowAttention, self).__init__()
        self.dim = dim # 64,可变
        self.window_size = window_size # 2,可变
        self.num_heads = num_heads # 8,可变
        self.scale = (dim // num_heads) ** -0.5 # 0.3535533905932738,dim // num_heads 进行平方根的倒数计算，一种经验做法
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # qkv 表示的是查询（query）、键（key）和值（value）这三种向量，通常会共享一个线性变换来计算它们，3 * C 是查询、键和值三者的通道数（通常是三倍的原始通道数）
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)

        # 这里是PyTorch版本中的初始化部分，相当于TensorFlow中的build
        num_window_elements = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
        # 定义相对位置偏置的权重
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_window_elements, self.num_heads))

        # 计算相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_matrix = torch.meshgrid(coords_h, coords_w)
        coords = torch.stack(coords_matrix)
        coords_flatten = coords.view(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0)
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.relative_position_index = relative_coords.sum(-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads) # self.num_heads：注意力头数（multi-head attention 中的头数），C // self.num_heads：每个头的通道数。由于原始通道数 C 会被平分给每个头，所以每个头的通道数是 C // self.num_heads。
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, B_, self.num_heads,N ,C // self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale #shape = (B_, self.num_heads,N, N) 精华，计算两个向量相似度，数值越大越相似（q⋅k=∥q∥∥k∥cos(θ)=坐标直接计算点积），self.scale是缩放因子，防止数值过大导致softmax操作的梯度消失或梯度爆炸问题。

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = self.relative_position_index.view(-1)
        # 获取相对位置偏置
        relative_position_bias = self.relative_position_bias_table[relative_position_index_flat]
        relative_position_bias = relative_position_bias.view(num_window_elements, num_window_elements, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1)
        attn = attn + relative_position_bias.unsqueeze(0)  # 加上相对位置偏置

        if mask is not None:
            nW = mask.shape[0] # mask的shape为（1，4，4），表示注意力窗口window_size为2时，分割了两行两列窗口
            mask = mask.to(attn.device)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 2, 1, 3).reshape(B_, N, C)
        x = self.proj(x)
        x = self.attn_drop(x)
        return x

class SwinTransformer(nn.Module):
    def __init__(self, dim, num_patch, num_heads, window_size=7, shift_size=0, num_mlp=1024,
                 qkv_bias=True, dropout_rate=0.):
        super().__init__()

        self.dim = dim
        self.num_patch = num_patch
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_mlp = num_mlp

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=(window_size, window_size),
                                    num_heads=num_heads, qkv_bias=qkv_bias, dropout_rate=dropout_rate)
        self.drop_path = DropPath(dropout_rate)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, num_mlp),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_mlp, dim),
            nn.Dropout(dropout_rate),
        )

        if min(self.num_patch) < self.window_size: # 防止不能被整除
            self.shift_size = 0
            self.window_size = min(self.num_patch)

        self.H, self.W = num_patch
        self.attn_mask = self.create_mask(self.H, self.W, window_size, shift_size) if shift_size > 0 else None

    def create_mask(self, H, W, window_size, shift_size):
        img_mask = torch.zeros((1, H, W, 1)) # 生成shape为（1，H，W，1）大小的0向量。H W 为前面PatchExtract抓取patchs行与列个数
        cnt = 0
        h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, window_size)
        mask_windows = mask_windows.view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.H, self.W
        x_skip = x
        x = self.norm1(x) # 标准化
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C) # shape为[16, 2, 2, 64]
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = self.drop_path(x) + x_skip
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x) + x_skip
        return x

class PatchMerging(nn.Module):
    def __init__(self, num_patch, embed_dim):
        super(PatchMerging, self).__init__()
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.linear_trans = nn.Linear(4 * embed_dim, 2 * embed_dim, bias=False)

    def forward(self, x):
        height, width = self.num_patch
        B, N, C = x.shape
        x = x.view(B, height, width, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat((x0, x1, x2, x3), dim=-1)
        x = x.view(B, (height // 2) * (width // 2), 4 * C)
        return self.linear_trans(x)
    # TODO 测试模块

def conv_layer_vgg(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer

def vgg_conv_block_vgg(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [conv_layer_vgg(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    #layers += [tnn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return nn.Sequential(*layers)


def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer

def get_freq_indices(method):
    # 获得分量排名的坐标
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        # eg, c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
        # eg, dct_h = c2wh[planes], dct_w = c2wh[planes]
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w
        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)  # 16
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7
        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)
        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)

class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        assert len(mapper_x) == len(mapper_y)
        # assert channel % len(mapper_x) == 0
        self.num_freq = len(mapper_x)
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape
        #print(x.shape)  # [24, 72, 56, 56]
        #print(self.weight.shape)  # [288, 56, 56]
        x = x * self.weight
        result = torch.sum(x, dim=[2, 3])  # 在空间维度上求和
        return result
    def build_filter(self, pos, freq, POS):  # 对应i/j, h/w, H/W
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)  # 基函数公式的一半
        if freq == 0:
            # 对应gap的形式
            return result
        else:
            return result * math.sqrt(2)
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)  # 对于每一个BATCH都是相同的
        # c_part = channel // len(mapper_x)  # 每一份的通道长度
        c_part = 1
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)
                    # dct_filter[i: (i + 1), t_x, t_y] = self.build_filter(t_x, u_x,tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
        return dct_filter

# @torchsnooper.snoop()
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        # torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        # torch.nn.init.zeros_(self.to_qkv.bias)
        self.nn1 = nn.Linear(dim, dim)
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask
        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper
        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))
    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)  # go to attention
            x = mlp(x)  # go to MLP_Block
        return x

class GWTransformer(nn.Module):
    def __init__(self, num_tokens=4, dim=128, channelsNum=64, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        super(GWTransformer, self).__init__()
        # Tokenization
        self.L = num_tokens
        self.cT = dim
        self.token_wA = nn.Parameter(torch.empty(1, self.L, channelsNum),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, channelsNum, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)
        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()
    def forward(self, x):
        # 参考SSFTT
        x = rearrange(x, 'b c h w -> b (h w) c')
        #print("X2", x.size())
        wa = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        #print("wa", wa.size())
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)
        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        trans = self.transformer(x)  # main game
        x = trans + x
        return x

# 神经网络搭建
# TODO 测试网络
class NEW(nn.Module):
    def __init__(self, num_classes=6):
        super(NEW, self).__init__()

        # BrunswickS2
        self.s2_1 = nn.Sequential(
            nn.Conv2d(in_channels= modes_number[0] + modes_number[3], out_channels=32, kernel_size=1, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s2_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s2_3 = nn.MaxPool2d(kernel_size=1)

        self.s2_4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s2_5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s2_6 = nn.MaxPool2d(kernel_size=1)

        self.s2_7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s2_8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s2_9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s2_10 = nn.MaxPool2d(kernel_size=1)

        self.s2_11 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s2_12 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s2_13 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s2_14 = nn.MaxPool2d(kernel_size=1)

        self.s2_15 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s2_16 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s2_17 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s2_18 = nn.MaxPool2d(kernel_size=1)

        # BrunswickS1
        self.s1_1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(2, 2, 2), padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s1_2 = nn.BatchNorm3d(num_features=64)
        self.s1_3 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s1_4 = nn.Sequential(
            nn.Conv2d(in_channels=64*(cumulative_modes[2]-cumulative_modes[0]), out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s1_5 = nn.MaxPool2d(kernel_size=1)

        # 第一次合并
        self.concate_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=32, kernel_size=1, padding='same'),
        )
        self.concate_2 = nn.BatchNorm2d(num_features=32)
        self.concate_3 = nn.ReLU(inplace=True)
        self.concate_4 = nn.AvgPool2d(kernel_size=1)
        self.concate_5 = nn.Flatten()
        self.concate_6 = nn.Sequential(
            nn.Linear(in_features=800, out_features=100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=100, out_features=50),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4)
        )

        # DEM
        # DEM超参数
        self.Probability = 0.5
        self.DEM_patch_size = 3
        self.num_patch = (PPsize + 1)// self.DEM_patch_size
        self.embed_dim = 64
        # DEM网络层
        self.dem_1 = T.RandomCrop(size = (PPsize, PPsize))
        self.dem_2 = T.RandomHorizontalFlip(p = self.Probability) # p=0.5默认50%概率
        self.dem_3 = PatchExtract(patch_size = (self.DEM_patch_size, self.DEM_patch_size)) # DEM抓取patchs大小(DEM_patch_size, DEM_patch_size)
        self.dem_4 = PatchEmbedding(self.DEM_patch_size,self.num_patch * self.num_patch,self.embed_dim) # 每个patchs升维到embed_dim，并进行位置编码
        self.dem_5 = SwinTransformer(   dim = self.embed_dim, # 升维度长度
                                        num_patch = (self.num_patch, self.num_patch), # (5+1)/3=2 (5+1)/3=2 抓取patchs的个数
                                        num_heads = 8, # 8  Attention heads 注意力头，需要可以整除self.embed_dim
                                        window_size = 2, # 1 注意力窗口大小，不能大于patchs
                                        shift_size = 0, # 移动窗口大小，打破窗口间信息障壁，第一次使用0
                                        num_mlp = 256, # 256 MLP layer size
                                        qkv_bias = True, # qkv偏置，将嵌入补丁转换为具有可学习的附加值的查询、键和值。
                                        dropout_rate = 0.03)
        self.dem_6 = SwinTransformer(   dim = self.embed_dim, # 升维度长度
                                        num_patch = (self.num_patch, self.num_patch), # (5+1)/2=3 (5+1)/2=3 抓取patchs的个数
                                        num_heads = 8, # 8  Attention heads 注意力头
                                        window_size = 2, # 1 注意力窗口大小，不能大于patchs
                                        shift_size = 1, # 移动窗口大小，打破窗口间信息障壁
                                        num_mlp = 256, # 256 MLP layer size
                                        qkv_bias = True, # qkv偏置，将嵌入补丁转换为具有可学习的附加值的查询、键和值。
                                        dropout_rate = 0.03)
        self.dem_7 = PatchMerging((self.num_patch, self.num_patch), embed_dim=self.embed_dim)
        self.dem_8 = nn.Linear(in_features=128, out_features=50)

        # 输出层
        self.output_layer = nn.Linear(100, out_class)

    def forward(self, x):
        # Sentinel-2 波段
        s2_ndvi = x[:, 0:cumulative_modes[0], :, :]
        s2_band = x[:, cumulative_modes[2]: cumulative_modes[3], :, :]
        s2 = torch.cat((s2_ndvi, s2_band), dim = 1)
        s2 = self.s2_1(s2)
        s2 = self.s2_2(s2)
        s2 = self.s2_3(s2)
        s2 = self.s2_4(s2)
        s2 = self.s2_5(s2)
        s2 = self.s2_6(s2)
        s2 = self.s2_7(s2)
        s2 = self.s2_8(s2)
        s2 = self.s2_9(s2)
        s2 = self.s2_10(s2)
        s2 = self.s2_11(s2)
        s2 = self.s2_12(s2)
        s2 = self.s2_13(s2)
        s2 = self.s2_14(s2)
        s2 = self.s2_15(s2)
        s2 = self.s2_16(s2)
        s2 = self.s2_17(s2)
        s2 = self.s2_18(s2)

        # Sentinel-1 波段
        s1 = x[:, cumulative_modes[0]: cumulative_modes[2], :, :]
        s1 = torch.unsqueeze(s1, 1)
        s1 = self.s1_1(s1)
        s1 = self.s1_2(s1)
        s1 = self.s1_3(s1)
        s1 = s1.reshape(s1.size(0), s1.size(1) * s1.size(2), s1.size(3), s1.size(4))
        s1 = self.s1_4(s1)
        s1 = self.s1_5(s1)

        # 第一次合并
        concate1 = torch.cat((s2, s1), dim=1)
        concate1 = self.concate_1(concate1)
        concate1 = self.concate_2(concate1)
        concate1 = self.concate_3(concate1)
        concate1 = self.concate_4(concate1)
        concate1 = self.concate_5(concate1)
        concate1 = self.concate_6(concate1)

        # dem 波段
        dem = x[:, cumulative_modes[3]: cumulative_modes[4], :, :]
        dem = self.dem_1(dem)
        dem = self.dem_2(dem)
        dem = self.dem_3(dem)
        dem = dem.transpose(1,2)
        dem = self.dem_4(dem)
        dem = self.dem_5(dem)
        dem = self.dem_6(dem)
        dem = self.dem_7(dem)
        dem = F.adaptive_avg_pool1d(dem.permute(0, 2, 1), 1)
        dem = dem.squeeze(-1)
        dem = self.dem_8(dem)

        # 第二次合并
        concate2 = torch.cat((concate1, dem), dim=1)

        # 结果输出
        out = self.output_layer(concate2)
        output = F.softmax(out, dim=1)

        return output

# TODO 测试网络

# TODO HRN正式版本
class HRN(nn.Module):
    def __init__(self, dataset):
        """
        modes_number: 模态通道数列表，如[2,5,1]表示3个模态，通道数分别为2,5,1
        out_class: 分类任务的类别数
        """
        super(HRN, self).__init__()
        self.dataset = dataset
        self.modes_number = modes_number
        self.num_modalities = len(modes_number)
        self.out_class = out_class

        # 为每个模态创建独立的特征提取分支
        self.modalities = nn.ModuleList()
        for i, channels in enumerate(modes_number):

            # 每个模态的特征提取模块
            modality = nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=30, kernel_size=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(30),
            )
            setattr(self, f'modality_{i}', modality)

            # 每个模态的3D卷积层
            conv = nn.Sequential(
                nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3), padding=(0, 1, 1)),
                nn.BatchNorm3d(8),
                nn.ReLU(inplace=True),
            )
            setattr(self, f'conv1_{i}', conv)

            conv = nn.Sequential(
                nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3), padding=(0, 1, 1)),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
            )
            setattr(self, f'conv2_{i}', conv)

            conv = nn.Sequential(
                nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), padding=(0, 1, 1)),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
            )
            setattr(self, f'conv3_{i}', conv)

            # 每个模态的压缩卷积层
            conv = nn.Sequential(
                nn.Conv2d(in_channels=576, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            setattr(self, f'conv4_{i}', conv)

            # 每个模态的全连接层
            fc1 = nn.Sequential(
                nn.Linear(64 * get_PPsize() * get_PPsize(), 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.4),
            )
            setattr(self, f'fc1_{i}', fc1)

            fc2 = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.4),
            )
            setattr(self, f'fc2_{i}', fc2)

            classifier = nn.Linear(128, out_class)
            setattr(self, f'classifier_{i}', classifier)

        # 注意力机制
        #multi_Attention
        #1.
        self.se = SEAttention(channel=64*len(modes_number), reduction=8)
        #2
        self.bam = BAMBlock(channel=64*len(modes_number), reduction=16, dia_val=1)
        #3
        self.eca = ECAAttention(kernel_size=3)
        #4
        self.fac = MultiSpectralAttentionLayer(64*len(modes_number), 5, 5,  reduction=16, freq_sel_method = 'top16')
        #5.
        self.cbam = CBAMBlock(channel=64*len(modes_number), reduction=16, kernel_size=5)
        self.cbam = CBAMBlock(channel=64 * len(modes_number), reduction=16, kernel_size=5)

        # 注意力输出
        self.cascadeFC1 = nn.Sequential(
            nn.Linear(64*len(modes_number) * get_PPsize() * get_PPsize(), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )

        self.cascadeFC2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )
        self.cascadeclassifier = nn.Linear(128, out_class)

        # 模态注意力权重（可学习参数）
        # self.weights = nn.Parameter(torch.ones(self.num_modalities) / self.num_modalities)

    def forward(self, x, weights):
        # 数据与参数输入
        weights_tensor = torch.from_numpy(weights).float().to(device)

        # 消融实验，取消注释下面一行权重默认为1
        weights_tensor = torch.ones_like(weights_tensor)

        features = []
        outs = []

        # 按模态分割输入并分别处理
        start = 0
        for i, channels in enumerate(self.modes_number):
            end = start + channels
            modality_input = x[:, start:end, :, :]
            start = end

            # 通过对应模态的分支
            fe = getattr(self, f'modality_{i}')(modality_input)
            fe = torch.unsqueeze(fe, 1)
            conv = getattr(self, f'conv1_{i}')(fe)
            conv = getattr(self, f'conv2_{i}')(conv)
            conv = getattr(self, f'conv3_{i}')(conv)
            conv = torch.reshape(conv, (conv.shape[0], -1, conv.shape[3], conv.shape[4]))
            conv = getattr(self, f'conv4_{i}')(conv)
            features.append(conv)
            conv = torch.reshape(conv, (conv.shape[0], -1))
            fc1 = getattr(self, f'fc1_{i}')(conv)
            fc2 = getattr(self, f'fc2_{i}')(fc1)
            out = getattr(self, f'classifier_{i}')(fc2)

            outs.append(out)

        # 模态融合
        cascade = torch.cat([features[i] for i in range(self.num_modalities)], dim=1)
        # cascade = self.se(cascade) #内部无残差，开封第二最厉害解决
        # cascade = self.bam(cascade) #内部有残差，uav厉害解决
        # cascade = self.eca(cascade) #内部无残差，uav最厉害解决，开封第三厉害
        # cascade = cascade + self.eca(cascade) #内部无残差，厉害解决
        # cascade = self.fac(cascade)
        cascade = self.cbam(cascade)
        cascade = torch.reshape(cascade, (cascade.shape[0], -1))
        cascade = self.cascadeFC1(cascade)
        cascade = self.cascadeFC2(cascade)
        cascade_out = self.cascadeclassifier(cascade)

        # 模态加权
        outputs = torch.stack(outs, dim=1)
        outputs_weighted = torch.zeros_like(outputs[:, 0, :])
        multi_modal = []
        # 遍历每个模态
        for i in range(self.num_modalities):
            # 将权重张量广播到每个batch
            weight_broadcasted = weights_tensor[i, :].unsqueeze(0)  # 形状变为 [1, 6]
            # 将每个模态的输出结果乘以对应的权重
            outputs_weighted_i = outputs[:, i, :] * weight_broadcasted
            multi_modal.append(outputs_weighted_i)
            outputs_weighted += outputs[:, i, :] * weight_broadcasted

        # 消融实验，注释下面四行
        cascade_weight_broadcasted = weights_tensor[self.num_modalities, :].unsqueeze(0)
        cascade_outputs_weighted = cascade_out * cascade_weight_broadcasted
        multi_modal.append(cascade_outputs_weighted)
        outputs_weighted += cascade_out * cascade_weight_broadcasted

        return outputs_weighted, multi_modal
# TODO HRN正式版本

# TODO HRN复杂写法（测试版本）
# class HRN(nn.Module):
# #     """
# #     Me
# #     """
#     def __init__(self, dataset, depth=1, heads=8, mlp_dim=8, dropout=0.1):
#         super(HRN, self).__init__()
#         self.dataset = dataset
#
#         self.FE = nn.Sequential(
#             nn.Conv2d(in_channels= modes_number[0], out_channels=30, kernel_size=(1, 1), stride=(1, 1)),
#             nn.BatchNorm2d(30),
#         )
#         self.conv1 = nn.Sequential(
#             # Notice:cause input shape is [N,C,D,H,W]，kernel_size here should be (D,H,W)
#             nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3), padding=(0, 1, 1)),
#             nn.BatchNorm3d(8),
#             nn.ReLU(inplace=True),
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3), padding=(0, 1, 1)),
#             nn.BatchNorm3d(16),
#             nn.ReLU(inplace=True),
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), padding=(0, 1, 1)),
#             nn.BatchNorm3d(32),
#             nn.ReLU(inplace=True),
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(in_channels=576, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )
#         self.FC1 = nn.Sequential(
#             nn.Linear(64 * get_PPsize() * get_PPsize(), 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.4),
#         )
#         self.FC2 = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.4),
#         )
#         self.classifier = nn.Linear(128, out_class)
#
#         #雷达模态
#         self.SARFE = nn.Sequential(
#             nn.Conv2d(in_channels= modes_number[1], out_channels=30, kernel_size=(1, 1), stride=(1, 1)),
#             nn.BatchNorm2d(30),
#         )
#         self.SARconv1 = nn.Sequential(
#             nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3), padding=(0, 1, 1)),
#             nn.BatchNorm3d(8),
#             nn.ReLU(inplace=True),
#         )
#         self.SARconv2 = nn.Sequential(
#             nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3), padding=(0, 1, 1)),
#             nn.BatchNorm3d(16),
#             nn.ReLU(inplace=True),
#         )
#         self.SARconv3 = nn.Sequential(
#             nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), padding=(0, 1, 1)),
#             nn.BatchNorm3d(32),
#             nn.ReLU(inplace=True),
#         )
#         self.SARconv4 = nn.Sequential(
#             nn.Conv2d(in_channels=576, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )
#         self.SARFC1 = nn.Sequential(
#             nn.Linear(64 * get_PPsize()* get_PPsize(), 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.4),
#         )
#         self.SARFC2 = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.4),
#         )
#         self.SARclassifier = nn.Linear(128, out_class)
#
#         #雷达模态2
#         self.SAR2FE = nn.Sequential(
#             nn.Conv2d(in_channels= modes_number[2], out_channels=30, kernel_size=(1, 1), stride=(1, 1)),
#             nn.BatchNorm2d(30),
#         )
#         self.SAR2conv1 = nn.Sequential(
#             nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3), padding=(0, 1, 1)),
#             nn.BatchNorm3d(8),
#             nn.ReLU(inplace=True),
#         )
#         self.SAR2conv2 = nn.Sequential(
#             nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3), padding=(0, 1, 1)),
#             nn.BatchNorm3d(16),
#             nn.ReLU(inplace=True),
#         )
#         self.SAR2conv3 = nn.Sequential(
#             nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), padding=(0, 1, 1)),
#             nn.BatchNorm3d(32),
#             nn.ReLU(inplace=True),
#         )
#         self.SAR2conv4 = nn.Sequential(
#             nn.Conv2d(in_channels=576, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )
#         self.SAR2FC1 = nn.Sequential(
#             nn.Linear(64 * get_PPsize()* get_PPsize(), 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.4),
#         )
#         self.SAR2FC2 = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.4),
#         )
#         self.SAR2classifier = nn.Linear(128, out_class)
#
#         #multi_Attention
#         #1.
#         self.se = SEAttention(channel=64*len(modes_number), reduction=8)
#         #2
#         self.bam = BAMBlock(channel=64*len(modes_number), reduction=16, dia_val=1)
#         #3
#         self.eca = ECAAttention(kernel_size=3)
#         #4
#         self.fac = MultiSpectralAttentionLayer(64*len(modes_number), 5, 5,  reduction=16, freq_sel_method = 'top16')
#         #5.
#         self.cbam = CBAMBlock(channel=64*len(modes_number), reduction=16, kernel_size=5)
#         # 注意力输出
#         self.cascadeFC1 = nn.Sequential(
#             nn.Linear(64*len(modes_number) * get_PPsize() * get_PPsize(), 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.4),
#         )
#         self.cascadeFC2 = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.4),
#         )
#         self.cascadeclassifier = nn.Linear(128, out_class)
#
#     def forward(self, x, weights):
#         #数据与参数输入
#         #print(weights)
#         weights_tensor = torch.from_numpy(weights).float().to(device)
#         #print(weights_tensor)
#         data1 = x[:, :modes_number[0], :, :] #Python 切片是左闭右开的
#         data2 = x[:, modes_number[0]:modes_number[0]+modes_number[1], :, :]
#         data3 = x[:, modes_number[0]+modes_number[1]:modes_number[0]+modes_number[1]+modes_number[2], :, :]
#
#         fe = self.FE(data1)
#         fe = torch.unsqueeze(fe, 1)
#         conv1 = self.conv1(fe)
#         conv2 = self.conv2(conv1)
#         conv3 = self.conv3(conv2)
#         conv3 = torch.reshape(conv3, (conv3.shape[0], -1, conv3.shape[3], conv3.shape[4]))
#         conv4 = self.conv4(conv3)
#
#         SARfe = self.SARFE(data2)
#         SARfe = torch.unsqueeze(SARfe, 1)
#         SARconv1 = self.SARconv1(SARfe)
#         SARconv2 = self.SARconv2(SARconv1)
#         SARconv3 = self.SARconv3(SARconv2)
#         SARconv3 = torch.reshape(SARconv3, (SARconv3.shape[0], -1, SARconv3.shape[3], SARconv3.shape[4]))
#         SARconv4 = self.SARconv4(SARconv3)
#
#         SARfe2 = self.SAR2FE(data3)
#         SARfe2 = torch.unsqueeze(SARfe2, 1)
#         SARconv12 = self.SAR2conv1(SARfe2)
#         SARconv22 = self.SAR2conv2(SARconv12)
#         SARconv32 = self.SAR2conv3(SARconv22)
#         SARconv32 = torch.reshape(SARconv32, (SARconv32.shape[0], -1, SARconv32.shape[3], SARconv32.shape[4]))
#         SARconv42 = self.SAR2conv4(SARconv32)
#
#         cascade = torch.cat((conv4, SARconv4, SARconv42), dim=1)
#         # cascade = self.se(cascade) #内部无残差，开封第二最厉害解决
#         # cascade = self.bam(cascade) #内部有残差，uav厉害解决
#         # cascade = self.eca(cascade) #内部无残差，uav最厉害解决，开封第三厉害
#         # cascade = cascade + self.eca(cascade) #内部无残差，厉害解决
#         # cascade = self.fac(cascade)
#         cascade = self.cbam(cascade)#内部有残差，开封最厉害
#
#         cascade = torch.reshape(cascade, (cascade.shape[0], -1))
#         cascade = self.cascadeFC1(cascade)
#         cascade = self.cascadeFC2(cascade)
#         out4 = self.cascadeclassifier(cascade)
#
#         conv4 = torch.reshape(conv4, (conv4.shape[0], -1))
#         fc1 = self.FC1(conv4)
#         fc2 = self.FC2(fc1)
#         out1 = self.classifier(fc2)
#
#         SARconv4 = torch.reshape(SARconv4, (SARconv4.shape[0], -1))
#         SARfc1 = self.SARFC1(SARconv4)
#         SARfc2 = self.SARFC2(SARfc1)
#         out2 = self.SARclassifier(SARfc2)
#
#         SARconv42 = torch.reshape(SARconv42, (SARconv42.shape[0], -1))
#         SARfc12 = self.SAR2FC1(SARconv42)
#         SARfc22 = self.SAR2FC2(SARfc12)
#         out3 = self.SAR2classifier(SARfc22)

#         # 模态加权
#         out1 = out1 * weights_tensor[0]
#         out2 = out2 * weights_tensor[1]
#         out3 = out3 * weights_tensor[2]
#
#         outputs = out1 + out2 + out3 + out4
#
#         return outputs, [out1, out2, out3, out4]
# TODO HRN复杂写法（测试版本）

class MultiModelCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(MultiModelCNN, self).__init__()

        # BrunswickS2
        self.s2_1 = nn.Sequential(
            nn.Conv2d(in_channels= modes_number[0] + modes_number[3], out_channels=32, kernel_size=1, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s2_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s2_3 = nn.MaxPool2d(kernel_size=1)

        self.s2_4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s2_5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s2_6 = nn.MaxPool2d(kernel_size=1)

        self.s2_7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s2_8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s2_9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s2_10 = nn.MaxPool2d(kernel_size=1)

        self.s2_11 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s2_12 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s2_13 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s2_14 = nn.MaxPool2d(kernel_size=1)

        self.s2_15 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s2_16 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s2_17 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s2_18 = nn.MaxPool2d(kernel_size=1)

        # BrunswickS1
        self.s1_1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(2, 2, 2), padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s1_2 = nn.BatchNorm3d(num_features=64)
        self.s1_3 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s1_4 = nn.Sequential(
            nn.Conv2d(in_channels=64*(cumulative_modes[2]-cumulative_modes[0]), out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.s1_5 = nn.MaxPool2d(kernel_size=1)

        # 第一次合并
        self.concate_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=32, kernel_size=1, padding='same'),
        )
        self.concate_2 = nn.BatchNorm2d(num_features=32)
        self.concate_3 = nn.ReLU(inplace=True)
        self.concate_4 = nn.AvgPool2d(kernel_size=1)
        self.concate_5 = nn.Flatten()
        self.concate_6 = nn.Sequential(
            nn.Linear(in_features=800, out_features=100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=100, out_features=50),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4)
        )

        # DEM
        # DEM超参数
        self.Probability = 0.5
        self.DEM_patch_size = 3
        self.num_patch = (PPsize + 1)// self.DEM_patch_size
        self.embed_dim = 64
        # DEM网络层
        self.dem_1 = T.RandomCrop(size = (PPsize, PPsize))
        self.dem_2 = T.RandomHorizontalFlip(p = self.Probability) # p=0.5默认50%概率
        self.dem_3 = PatchExtract(patch_size = (self.DEM_patch_size, self.DEM_patch_size)) # DEM抓取patchs大小(DEM_patch_size, DEM_patch_size)
        self.dem_4 = PatchEmbedding(self.DEM_patch_size,self.num_patch * self.num_patch,self.embed_dim) # 每个patchs升维到embed_dim，并进行位置编码
        self.dem_5 = SwinTransformer(   dim = self.embed_dim, # 升维度长度
                                        num_patch = (self.num_patch, self.num_patch), # (5+1)/3=2 (5+1)/3=2 抓取patchs的个数
                                        num_heads = 8, # 8  Attention heads 注意力头，需要可以整除self.embed_dim
                                        window_size = 2, # 1 注意力窗口大小，不能大于patchs
                                        shift_size = 0, # 移动窗口大小，打破窗口间信息障壁，第一次使用0
                                        num_mlp = 256, # 256 MLP layer size
                                        qkv_bias = True, # qkv偏置，将嵌入补丁转换为具有可学习的附加值的查询、键和值。
                                        dropout_rate = 0.03)
        self.dem_6 = SwinTransformer(   dim = self.embed_dim, # 升维度长度
                                        num_patch = (self.num_patch, self.num_patch), # (5+1)/2=3 (5+1)/2=3 抓取patchs的个数
                                        num_heads = 8, # 8  Attention heads 注意力头
                                        window_size = 2, # 1 注意力窗口大小，不能大于patchs
                                        shift_size = 1, # 移动窗口大小，打破窗口间信息障壁
                                        num_mlp = 256, # 256 MLP layer size
                                        qkv_bias = True, # qkv偏置，将嵌入补丁转换为具有可学习的附加值的查询、键和值。
                                        dropout_rate = 0.03)
        self.dem_7 = PatchMerging((self.num_patch, self.num_patch), embed_dim=self.embed_dim)
        self.dem_8 = nn.Linear(in_features=128, out_features=50)

        # 输出层
        self.output_layer = nn.Linear(100, out_class)

    def forward(self, x):
        # Sentinel-2 波段
        s2_ndvi = x[:, 0:cumulative_modes[0], :, :]
        s2_band = x[:, cumulative_modes[2]: cumulative_modes[3], :, :]
        s2 = torch.cat((s2_ndvi, s2_band), dim = 1)
        s2 = self.s2_1(s2)
        s2 = self.s2_2(s2)
        s2 = self.s2_3(s2)
        s2 = self.s2_4(s2)
        s2 = self.s2_5(s2)
        s2 = self.s2_6(s2)
        s2 = self.s2_7(s2)
        s2 = self.s2_8(s2)
        s2 = self.s2_9(s2)
        s2 = self.s2_10(s2)
        s2 = self.s2_11(s2)
        s2 = self.s2_12(s2)
        s2 = self.s2_13(s2)
        s2 = self.s2_14(s2)
        s2 = self.s2_15(s2)
        s2 = self.s2_16(s2)
        s2 = self.s2_17(s2)
        s2 = self.s2_18(s2)

        # Sentinel-1 波段
        s1 = x[:, cumulative_modes[0]: cumulative_modes[2], :, :]
        s1 = torch.unsqueeze(s1, 1)
        s1 = self.s1_1(s1)
        s1 = self.s1_2(s1)
        s1 = self.s1_3(s1)
        s1 = s1.reshape(s1.size(0), s1.size(1) * s1.size(2), s1.size(3), s1.size(4))
        s1 = self.s1_4(s1)
        s1 = self.s1_5(s1)

        # 第一次合并
        concate1 = torch.cat((s2, s1), dim=1)
        concate1 = self.concate_1(concate1)
        concate1 = self.concate_2(concate1)
        concate1 = self.concate_3(concate1)
        concate1 = self.concate_4(concate1)
        concate1 = self.concate_5(concate1)
        concate1 = self.concate_6(concate1)

        # dem 波段
        dem = x[:, cumulative_modes[3]: cumulative_modes[4], :, :]
        dem = self.dem_1(dem)
        dem = self.dem_2(dem)
        dem = self.dem_3(dem)
        dem = dem.transpose(1,2)
        dem = self.dem_4(dem)
        dem = self.dem_5(dem)
        dem = self.dem_6(dem)
        dem = self.dem_7(dem)
        dem = F.adaptive_avg_pool1d(dem.permute(0, 2, 1), 1)
        dem = dem.squeeze(-1)
        dem = self.dem_8(dem)

        # 第二次合并
        concate2 = torch.cat((concate1, dem), dim=1)

        # 结果输出
        out = self.output_layer(concate2)
        output = F.softmax(out, dim=1)

        return output
#@torchsnooper.snoop()
class CNN_1D(nn.Module):
    """
    Based on Paper:Chen, Y. Deep Feature Extraction and Classification of Hyperspectral Images Based on Convolutional Neural Networks. TGRS
    input shape:[N,C=1,L=spectral_channel]
    """
    def __init__(self, dataset):
        super(CNN_1D, self).__init__()
        self.dataset = dataset
        self.C2 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=6, kernel_size=(3,)),
            nn.BatchNorm1d(6),
            nn.ReLU(inplace=True),
            #nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.C4 = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=12, kernel_size=(3,)),
            nn.BatchNorm1d(12),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            #nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.C6 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=24, kernel_size=(3,)),
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            #nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.C8 = nn.Sequential(
            nn.Conv1d(in_channels=24, out_channels=48, kernel_size=(3,)),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),  # for IP ,p=0.3, else(PU and SV) is 0.1
            #nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(816, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )
        self.classifier = nn.Linear(128, out_class)
    def forward(self, x):
        #x = torch.squeeze(x, dim=2).transpose(1, 2)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        C2 = self.C2(x)
        C4 = self.C4(C2)
        last = self.C6(C4)
        last = self.C8(last)
        last = torch.reshape(last, (last.shape[0], -1))
        FC = self.fc(last)
        out = self.classifier(FC)
        return out

# @torchsnooper.snoop()
class CNN_2D(nn.Module):
    """
    Based on paper: Chen, Y. Deep Feature Extraction and Classification of Hyperspectral Images Based on Convolutional Neural Networks. TGRS
    Input shape:[N,C=spectral_channel,H=5,W=5]
    """
    def __init__(self, dataset):
        super(CNN_2D, self).__init__()
        self.dataset = dataset
        self.FE = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=get_in_planes(dataset), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(get_in_planes(dataset)),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=get_in_planes(dataset), out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

        )#最后  nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Dropout2d(p=0.5),
        )#nn.Dropout2d(p=0.5),的前面，nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
        )
        self.classifier = nn.Linear(get_fc_in(dataset, 'CNN_2D_new'), out_class)
    def forward(self, x):
        FE = self.FE(x)
        layer1 = self.layer1(FE)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer3 = torch.reshape(layer3, (layer3.shape[0], -1))
        # print(layer3.size())
        out = self.classifier(layer3)
        return out

# @torchsnooper.snoop()
class CNN_3D(nn.Module):
    """
     Based on: Chen, Y. Deep Feature Extraction and Classification of Hyperspectral Images Based on Convolutional Neural Networks. TGRS
    Input shape:[N,C=spectral_channel,H=5,W=5]
    """

    # 3D卷积大输入通道为五维，batch_size、通道数、帧数(每通道内)、高、宽
    # in_channels = 通道数 、 out_channels = 卷积核数  kernel_size = 小于 帧数、高、宽
    def __init__(self, dataset):
        super(CNN_3D, self).__init__()
        self.dataset = dataset
        out_channels = [32, 64, 128]
        if in_channel == 1:
            self.layer1 = nn.Sequential(
                nn.Conv3d(in_channels=1, out_channels=out_channels[0], kernel_size=(1, 3, 3), stride=(1, 1, 1),
                          padding=(0, 1, 1)),
                nn.BatchNorm3d(out_channels[0]),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # kernel_size=(2, 3, 3)
            )
        else:
            self.layer1 = nn.Sequential(
                nn.Conv3d(in_channels=1, out_channels=out_channels[0], kernel_size=(2, 3, 3), stride=(1, 1, 1),
                          padding=(0, 1, 1)),
                nn.BatchNorm3d(out_channels[0]),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # kernel_size=(2, 3, 3)
            )
        self.layer2 = nn.Sequential(
            nn.Conv3d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=(1, 2, 2), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(p=0.5),

        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(out_channels[2]),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.5),

        )
        if in_channel > 1:
            self.classifier = nn.Linear(128*(in_channel-1), out_class)
        else:
            self.classifier = nn.Linear(128, out_class)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer3 = torch.reshape(layer3, (layer3.shape[0], -1))
        # print(layer3.size())
        out = self.classifier(layer3)
        return out

# @torchsnooper.snoop()
class HybridSN(nn.Module):
    """
    Based on paper:HybridSN: Exploring 3-D-2-D CNN Feature Hierarchy for Hyperspectral Image Classification. IEEE Geoscience and Remote Sensing Letters
    Input shape:[N,C=spectral_channel,H=5,W=5]
    """
    def __init__(self, dataset):
        super(HybridSN, self).__init__()
        self.dataset = dataset
        self.FE = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=30, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(30),
        )
        self.conv1 = nn.Sequential(
            # Notice:cause input shape is [N,C,D,H,W]，kernel_size here should be (D,H,W)
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=576, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # get_fc_in(dataset, 'HybridSN')=64 * PPsize * PPsize
        self.FC1 = nn.Sequential(
            nn.Linear(64 * get_PPsize()* get_PPsize(), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )
        self.FC2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )
        self.classifier = nn.Linear(128, out_class)
    def forward(self, x):
        fe = self.FE(x)
        fe = torch.unsqueeze(fe, 1)
        conv1 = self.conv1(fe)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv3 = torch.reshape(conv3, (
        conv3.shape[0], -1, conv3.shape[3], conv3.shape[4]))
        conv4 = self.conv4(conv3)
        conv4 = torch.reshape(conv4, (conv4.shape[0], -1))
        # print(conv4.size())  #这里打印看下输入全连接层前feature map的大小,self.fc1 = nn.Linear(2000, 500)  # 输入通道数是2000，输出通道数是500
        fc1 = self.FC1(conv4)
        fc2 = self.FC2(fc1)
        # print(fc2.size())
        out = self.classifier(fc2)
        return out

# @torchsnooper.snoop()
class SSAN(nn.Module):
    """
    Based on paper: Sun, H. et.al,Spectral-Spatial Attention Network for Hyperspectral Image Classification. TGRS
    Input shape:[N,C=spectral_channel,H=5,W=5]
    """
    def __init__(self, dataset):
        super(SSAN, self).__init__()
        self.dataset = dataset
        self.spe1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.CF1 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=128, kernel_size=(in_channel, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.spe_AM1 = Attention_gate(gate_channels=get_SSAN_gate_channel(dataset), gate_depth=64)

        self.spa1 = nn.Sequential(
            nn.Conv2d(in_channels=get_SSAN_gate_channel(dataset), out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.spa_AM1 = Attention_gate(gate_channels=64, gate_depth=64)

        self.spa2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.spa_AM2 = Attention_gate(gate_channels=64, gate_depth=64)

        self.FC1 = nn.Sequential(
            nn.Linear(64*PPsize*PPsize, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.FC2 = nn.Linear(256, out_class)
    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        spe1 = self.spe1(x)
        CF1 = self.CF1(spe1)
        CF1 = torch.squeeze(CF1)
        # print(CF1.size())
        spe_AM1 = self.spe_AM1(CF1)
        spa1 = self.spa1(spe_AM1)
        spa_AM1 = self.spa_AM1(spa1)
        spa2 = self.spa2(spa_AM1)
        spa_AM2 = self.spa_AM2(spa2)
        spa_AM2 = torch.reshape(spa_AM2, (spa_AM2.shape[0], -1))
        FC1 = self.FC1(spa_AM2)
        out = self.FC2(FC1)
        return out

# @torchsnooper.snoop()
class pResNet(nn.Module):
    """
    Based on paper:Paoletti. Deep pyramidal residual networks for spectral-spatial hyperspectral image classification. TGRS
    Input shape:[N,C=spectral_channel,H=5,W=5]
    In source code, each layer have 3 bottlenecks, i change to 2 bottlenecks each layer, but still with 3 layer
    """
    def __init__(self, dataset):
        super(pResNet, self).__init__()
        self.dataset = dataset
        self.in_planes = get_in_planes(dataset)
        self.FE = nn.Sequential(
            nn.Conv2d(in_channel, self.in_planes, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(self.in_planes),
        )
        self.layer1 = nn.Sequential(
            Bottleneck_TPPP(self.in_planes, 43),
            Bottleneck_TPPP(43*4, 54),
        )
        self.reduce1 = Bottleneck_TPPP(54 * 4, 54, stride=2, downsample=nn.AvgPool2d((2, 2), (2, 2)))
        self.layer2 = nn.Sequential(
            Bottleneck_TPPP(54*4, 65),
            Bottleneck_TPPP(65*4, 76),
        )
        self.reduce2 = Bottleneck_TPPP(76*4, 76, stride=2, downsample=nn.AvgPool2d((2, 2), (2, 2)))
        self.layer3 = nn.Sequential(
            Bottleneck_TPPP(76*4, 87),
            Bottleneck_TPPP(87*4, 98),
        )
        self.avgpool = nn.AvgPool2d(get_avgpoosize(dataset))
        self.classifier = nn.Linear(98*4, out_class)
    def forward(self, x):
        FE = self.FE(x)  # 降维
        layer1 = self.layer1(FE)
        reduce1 = self.reduce1(layer1)
        layer2 = self.layer2(reduce1)
        reduce2 = self.reduce2(layer2)
        layer3 = self.layer3(reduce2)
        avg = self.avgpool(layer3)
        avg = avg.view(avg.size(0), -1)
        out = self.classifier(avg)
        return out

# @torchsnooper.snoop()
class DL_Shallow_Network(nn.Module):
    """
    Based on paper:HybridSN: Exploring 3-D-2-D CNN Feature Hierarchy for Hyperspectral Image Classification. IEEE Geoscience and Remote Sensing Letters
    Input shape:[N,C=spectral_channel,H=5,W=5]
    """
    def __init__(self, dataset):
        super(DL_Shallow_Network, self).__init__()
        self.dataset = dataset
        self.FE = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=30, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(30),
        )
        self.conv1 = nn.Sequential(
            # Notice:cause input shape is [N,C,D,H,W]，kernel_size here should be (D,H,W)
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=576, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # get_fc_in(dataset, 'HybridSN')=64 * PPsize * PPsize
        self.FC1 = nn.Sequential(
            nn.Linear(64 * get_PPsize()* get_PPsize(), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )
        self.FC2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )
        self.classifier = nn.Linear(128, out_class)

    def forward(self, x, rf, logdir):
        fe = self.FE(x)
        fe = torch.unsqueeze(fe, 1)
        conv1 = self.conv1(fe)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv3 = torch.reshape(conv3, (
        conv3.shape[0], -1, conv3.shape[3], conv3.shape[4]))
        conv4 = self.conv4(conv3)
        conv4 = torch.reshape(conv4, (conv4.shape[0], -1))
        # print(conv4.size())  #这里打印看下输入全连接层前feature map的大小,self.fc1 = nn.Linear(2000, 500)  # 输入通道数是2000，输出通道数是500
        fc1 = self.FC1(conv4)
        fc2 = self.FC2(fc1)

        fc2_np = fc2.cpu().detach().numpy()

        if rf>0:
            print('使用了机器学习。')
            loaded_model = joblib.load(os.path.join(logdir, 'random_forest_model.joblib'))
            columnss = [f'fc2_{i}' for i in range(fc2.shape[1])]
            dff = pd.DataFrame(np.vstack(fc2_np), columns=columnss)
            y_prob = loaded_model.predict_proba(dff)
            out = torch.Tensor(y_prob).to(device)
        else:
            out = self.classifier(fc2)
        return out, fc2

# @torchsnooper.snoop()
class SSRN(nn.Module):
    """
    Based on paper:Zhong, Z. Spectral-Spatial Residual Network for Hyperspectral Image Classification: A 3-D Deep Learning Framework. TGRS
    Input shape:[N,C=spectral_channel,H=5,W=5]
    """
    def __init__(self, dataset):
        super(SSRN, self).__init__()
        self.dataset = dataset
        self.FE1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=24, kernel_size=(7, 1, 1), stride=(2, 1, 1)),
            nn.BatchNorm3d(24),
        )
        self.spe_conv1 = nn.Sequential(
            nn.Conv3d(in_channels=24, out_channels=24, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0)),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=24, out_channels=24, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0)),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
        )
        self.spe_conv2 = nn.Sequential(
            nn.Conv3d(in_channels=24, out_channels=24, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0)),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=24, out_channels=24, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0)),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
        )
        self.CF = nn.Sequential(
            nn.Conv3d(in_channels=24, out_channels=128, kernel_size=(get_SSRN_channel(dataset), 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(128),
        )
        self.FE2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=24, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(24),
        )  # 5*5*128->3*3*24
        self.spa_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
        )

        # PPsize=5
        # self.avgpool = nn.AvgPool2d(kernel_size=3)
        # PPsize=9   9-2
        self.avgpool = nn.AvgPool2d(kernel_size=get_PPsize()-2)
        self.classifier = nn.Linear(24, out_class)
    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        FE1 = self.FE1(x)
        spe_conv1 = self.spe_conv1(FE1)
        spe_conv1_new = spe_conv1 + FE1
        spe_conv2 = self.spe_conv2(spe_conv1_new)
        spe_conv2_new = spe_conv2 + spe_conv1_new
        CF = self.CF(spe_conv2_new)
        CF = torch.squeeze(CF, dim=2)
        FE2 = self.FE2(CF)
        spa_conv1 = self.spa_conv1(FE2)
        spa_conv1_new = spa_conv1 + FE2
        # spa_conv2 = self.spa_conv2(spa_conv1_new)
        # spa_conv2_new = spa_conv2 + spa_conv1_new
        # print("spa_conv1_new.size",spa_conv1_new.size)
        avg = self.avgpool(spa_conv1_new)
        # print(avg.size())
        avg = torch.squeeze(avg)
        # print(avg.size())
        out = self.classifier(avg)
        return out

class VGG16(nn.Module):
    def __init__(self, dataset):
        super(VGG16, self).__init__()
        self.dataset = dataset

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block_vgg([in_channel, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        # self.layer1 = tnn.Sequential(
        #     tnn.Conv2d(5, 64, kernel_size=3, padding=1),
        #     tnn.BatchNorm2d(64),
        #     tnn.ReLU()
        # )
        self.layer2 = vgg_conv_block_vgg([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.layer3 = vgg_conv_block_vgg([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer4 = vgg_conv_block_vgg([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer5 = vgg_conv_block_vgg([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)

        # FC layers
        self.layer6 = vgg_fc_layer(5 * 5 * 512, 4096)
        self.layer7 = vgg_fc_layer(4096, 4096)

        # Final layer
        self.layer8 = nn.Linear(4096, out_class)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)
        out = vgg16_features.contiguous().view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        return out

if __name__ == "__main__":
    """
    open torchsnooper-->test the shape change of each model
    """






