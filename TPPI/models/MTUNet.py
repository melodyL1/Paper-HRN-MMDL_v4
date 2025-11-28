#!/usr/bin/env python
# -*- coding:utf-8 -*-
import math
from einops import rearrange
from TPPI.models.utils import *
from TPPI.models.SSFTTnet import GWTransformer

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

class ConvBNReLU(nn.Module):
    def __init__(self,
                 c_in,
                 c_out,
                 kernel_size,
                 stride=1,
                 padding=1,
                 activation=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(c_in,
                              c_out,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(cin, cout, 3, 1, padding=1),
            ConvBNReLU(cout, cout, 3, stride=1, padding=1, activation=False))
        self.conv1 = nn.Conv2d(cout, cout, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.conv(x)
        h = x
        x = self.conv1(x)
        x = self.bn(x)
        x = h + x
        x = self.relu(x)
        return x


class U_encoder(nn.Module):
    def __init__(self):
        super(U_encoder, self).__init__()
        self.res1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.res2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.res3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        x = self.res1(x)
        features.append(x)  # (224, 224, 64)
        x = self.pool1(x)

        x = self.res2(x)
        features.append(x)  # (112, 112, 128)
        x = self.pool2(x)

        x = self.res3(x)
        features.append(x)  # (56, 56, 256)
        x = self.pool3(x)

        return x, features


class U_decoder(nn.Module):
    def __init__(self):
        super(U_decoder, self).__init__()
        self.trans1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.res1 = DoubleConv(512, 256)
        self.trans2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.res2 = DoubleConv(256, 128)
        self.trans3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.res3 = DoubleConv(128, 64)

    def forward(self, x, feature):

        x = self.trans1(x)  # (56, 56, 256)
        x = torch.cat((feature[2], x), dim=1)
        x = self.res1(x)  # (56, 56, 256)
        x = self.trans2(x)  # (112, 112, 128)
        x = torch.cat((feature[1], x), dim=1)
        x = self.res2(x)  # (112, 112, 128)
        x = self.trans3(x)  # (224, 224, 64)
        x = torch.cat((feature[0], x), dim=1)
        x = self.res3(x)
        return x


class MEAttention(nn.Module):
    def __init__(self, dim, configs):
        super(MEAttention, self).__init__()
        self.num_heads = configs["head"]
        self.coef = 4
        self.query_liner = nn.Linear(dim, dim * self.coef)
        self.num_heads = self.coef * self.num_heads
        self.k = 256 // self.coef
        self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k)
        self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads)

        self.proj = nn.Linear(dim * self.coef, dim)

    def forward(self, x):
        B, N, C = x.shape
        x = self.query_liner(x)
        x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1,
                                                     3)  #(1, 32, 225, 32)

        attn = self.linear_0(x)

        attn = attn.softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))

        x = self.linear_1(attn).permute(0, 2, 1, 3).reshape(B, N, -1)

        x = self.proj(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, configs, axial=False):
        super(Attention, self).__init__()
        self.axial = axial
        self.dim = dim
        self.num_head = configs["head"]
        self.attention_head_size = int(self.dim / configs["head"])
        self.all_head_size = self.num_head * self.attention_head_size

        self.query_layer = nn.Linear(self.dim, self.all_head_size)
        self.key_layer = nn.Linear(self.dim, self.all_head_size)
        self.value_layer = nn.Linear(self.dim, self.all_head_size)

        self.out = nn.Linear(self.dim, self.dim)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_head, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, x):
        # first row and col attention
        if self.axial:
            # row attention (single head attention)
            b, h, w, c = x.shape
            mixed_query_layer = self.query_layer(x)
            mixed_key_layer = self.key_layer(x)
            mixed_value_layer = self.value_layer(x)

            query_layer_x = mixed_query_layer.view(b * h, w, -1)
            key_layer_x = mixed_key_layer.view(b * h, w, -1).transpose(-1, -2)
            attention_scores_x = torch.matmul(query_layer_x,
                                              key_layer_x)  # (b*h, w, w, c)
            attention_scores_x = attention_scores_x.view(b, -1, w,
                                                         w)  # (b, h, w, w)

            # col attention  (single head attention)
            query_layer_y = mixed_query_layer.permute(0, 2, 1,
                                                      3).contiguous().view(
                                                          b * w, h, -1)
            key_layer_y = mixed_key_layer.permute(
                0, 2, 1, 3).contiguous().view(b * w, h, -1).transpose(-1, -2)
            attention_scores_y = torch.matmul(query_layer_y,
                                              key_layer_y)  # (b*w, h, h, c)
            attention_scores_y = attention_scores_y.view(b, -1, h,
                                                         h)  # (b, w, h, h)

            return attention_scores_x, attention_scores_y, mixed_value_layer

        else:

            mixed_query_layer = self.query_layer(x)
            mixed_key_layer = self.key_layer(x)
            mixed_value_layer = self.value_layer(x)

            query_layer = self.transpose_for_scores(mixed_query_layer).permute(
                0, 1, 2, 4, 3, 5).contiguous()  # (b, p, p, head, n, c)
            key_layer = self.transpose_for_scores(mixed_key_layer).permute(
                0, 1, 2, 4, 3, 5).contiguous()
            value_layer = self.transpose_for_scores(mixed_value_layer).permute(
                0, 1, 2, 4, 3, 5).contiguous()

            attention_scores = torch.matmul(query_layer,
                                            key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(
                self.attention_head_size)
            atten_probs = self.softmax(attention_scores)

            context_layer = torch.matmul(
                atten_probs, value_layer)  # (b, p, p, head, win, h)
            context_layer = context_layer.permute(0, 1, 2, 4, 3,
                                                  5).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (
                self.all_head_size, )
            context_layer = context_layer.view(*new_context_layer_shape)
            attention_output = self.out(context_layer)

        return attention_output


class WinAttention(nn.Module):
    def __init__(self, configs, dim):
        super(WinAttention, self).__init__()
        self.window_size = configs["win_size"]
        self.attention = Attention(dim, configs)

    def forward(self, x):
        b, n, c = x.shape
        h, w = int(np.sqrt(n)), int(np.sqrt(n))
        x = x.permute(0, 2, 1).contiguous().view(b, c, h, w)
        if h % self.window_size != 0:
            right_size = h + self.window_size - h % self.window_size
            new_x = torch.zeros((b, c, right_size, right_size))
            new_x[:, :, 0:x.shape[2], 0:x.shape[3]] = x[:]
            new_x[:, :, x.shape[2]:,
                  x.shape[3]:] = x[:, :, (x.shape[2] - right_size):,
                                   (x.shape[3] - right_size):]
            x = new_x
            b, c, h, w = x.shape
        x = x.view(b, c, h // self.window_size, self.window_size,
                   w // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 3, 5,
                      1).contiguous().view(b, h // self.window_size,
                                           w // self.window_size,
                                           self.window_size * self.window_size,
                                           c).cuda()
        x = self.attention(x)  # (b, p, p, win, h)
        return x


class DlightConv(nn.Module):
    def __init__(self, dim, configs):
        super(DlightConv, self).__init__()
        self.linear = nn.Linear(dim, configs["win_size"] * configs["win_size"])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        h = x
        avg_x = torch.mean(x, dim=-2)  # (b, n, n, 1, h)
        x_prob = self.softmax(self.linear(avg_x))  # (b, n, n, win)

        x = torch.mul(h,
                      x_prob.unsqueeze(-1))  # (b, p, p, 16, h) (b, p, p, 16)
        x = torch.sum(x, dim=-2)  # (b, n, n, 1, h)
        return x


class GaussianTrans(nn.Module):
    def __init__(self):
        super(GaussianTrans, self).__init__()
        self.bias = nn.Parameter(-torch.abs(torch.randn(1)))
        self.shift = nn.Parameter(torch.abs(torch.randn(1)))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x, atten_x_full, atten_y_full, value_full = x  # atten_x_full(b, h, w, w, c)   atten_y_full(b, w, h, h, c) value_full(b, h, w, c)
        new_value_full = torch.zeros_like(value_full)

        for r in range(x.shape[1]):  # row
            for c in range(x.shape[2]):  # col
                atten_x = atten_x_full[:, r, c, :]  # (b, w)
                atten_y = atten_y_full[:, c, r, :]  # (b, h)

                dis_x = torch.tensor([(h - c)**2 for h in range(x.shape[2])
                                      ]).cuda()  # (b, w)
                dis_y = torch.tensor([(w - r)**2 for w in range(x.shape[1])
                                      ]).cuda()  # (b, h)

                dis_x = -(self.shift * dis_x + self.bias).cuda()
                dis_y = -(self.shift * dis_y + self.bias).cuda()

                atten_x = self.softmax(dis_x + atten_x)
                atten_y = self.softmax(dis_y + atten_y)

                new_value_full[:, r, c, :] = torch.sum(
                    atten_x.unsqueeze(dim=-1) * value_full[:, r, :, :] +
                    atten_y.unsqueeze(dim=-1) * value_full[:, :, c, :],
                    dim=-2)
        return new_value_full


class CSAttention(nn.Module):
    def __init__(self, dim, configs):
        super(CSAttention, self).__init__()
        self.win_atten = WinAttention(configs, dim)
        self.dlightconv = DlightConv(dim, configs)
        self.global_atten = Attention(dim, configs, axial=True)
        self.gaussiantrans = GaussianTrans()
        #self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        #self.maxpool = nn.MaxPool2d(2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=4)
        self.queeze = nn.Conv2d(2 * dim, dim, 1)

    def forward(self, x):
        '''

        :param x: size(b, n, c)
        :return:
        '''
        origin_size = x.shape
        _, origin_h, origin_w, _ = origin_size[0], int(np.sqrt(
            origin_size[1])), int(np.sqrt(origin_size[1])), origin_size[2]
        x = self.win_atten(x)  # (b, p, p, win, h)
        b, p, p, win, c = x.shape
        h = x.view(b, p, p, int(np.sqrt(win)), int(np.sqrt(win)),
                   c).permute(0, 1, 3, 2, 4, 5).contiguous()
        h = h.view(b, p * int(np.sqrt(win)), p * int(np.sqrt(win)),
                   c).permute(0, 3, 1, 2).contiguous()  # (b, c, h, w)

        x = self.dlightconv(x)  # (b, n, n, h)
        atten_x, atten_y, mixed_value = self.global_atten(
            x)  # (atten_x, atten_y, value)
        gaussian_input = (x, atten_x, atten_y, mixed_value)
        x = self.gaussiantrans(gaussian_input)  # (b, h, w, c)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.up(x)
        x = self.queeze(torch.cat((x, h), dim=1)).permute(0, 2, 3,
                                                          1).contiguous()
        x = x[:, :origin_h, :origin_w, :].contiguous()
        x = x.view(b, -1, c)

        return x


class EAmodule(nn.Module):
    def __init__(self, dim):
        super(EAmodule, self).__init__()
        self.SlayerNorm = nn.LayerNorm(dim, eps=1e-6)
        self.ElayerNorm = nn.LayerNorm(dim, eps=1e-6)
        self.CSAttention = CSAttention(dim, configs)
        self.EAttention = MEAttention(dim, configs)

    def forward(self, x):
        h = x  # (B, N, H)
        x = self.SlayerNorm(x)

        x = self.CSAttention(x)  # padding 到right_size
        x = h + x

        h = x
        x = self.ElayerNorm(x)

        x = self.EAttention(x)
        x = h + x

        return x


class DecoderStem(nn.Module):
    def __init__(self):
        super(DecoderStem, self).__init__()
        self.block = U_decoder()

    def forward(self, x, features):
        x = self.block(x, features)
        return x


class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()
        self.model = U_encoder()
        self.trans_dim = ConvBNReLU(256, 256, 1, 1, 0)  #out_dim, model_dim
        self.position_embedding = nn.Parameter(torch.zeros((1, 784, 256)))

    def forward(self, x):

        x, features = self.model(x)  # (1, 512, 28, 28)
        x = self.trans_dim(x)  # (B, C, H, W) (1, 256, 28, 28)
        x = x.flatten(2)  # (B, H, N)  (1, 256, 28*28)
        x = x.transpose(-2, -1)  #  (B, N, H)
        x = x + self.position_embedding
        return x, features  #(B, N, H)


class encoder_block(nn.Module):
    def __init__(self, dim):
        super(encoder_block, self).__init__()
        self.block = nn.ModuleList([
            EAmodule(dim),
            EAmodule(dim),
            ConvBNReLU(dim, dim * 2, 2, stride=2, padding=0)
        ])

    def forward(self, x):
        x = self.block[0](x)
        x = self.block[1](x)
        B, N, C = x.shape
        h, w = int(np.sqrt(N)), int(np.sqrt(N))
        x = x.view(B, h, w, C).permute(0, 3, 1,
                                       2)  # (1, 256, 28, 28) B, C, H, W
        skip = x
        x = self.block[2](x)  # (14, 14, 256)
        return x, skip


class decoder_block(nn.Module):
    def __init__(self, dim, flag):
        super(decoder_block, self).__init__()
        self.flag = flag
        if not self.flag:
            self.block = nn.ModuleList([
                nn.ConvTranspose2d(dim,
                                   dim // 2,
                                   kernel_size=2,
                                   stride=2,
                                   padding=0),
                nn.Conv2d(dim, dim // 2, kernel_size=1, stride=1),
                EAmodule(dim // 2),
                EAmodule(dim // 2)
            ])
        else:
            self.block = nn.ModuleList([
                nn.ConvTranspose2d(dim,
                                   dim // 2,
                                   kernel_size=2,
                                   stride=2,
                                   padding=0),
                EAmodule(dim),
                EAmodule(dim)
            ])

    def forward(self, x, skip):
        if not self.flag:
            x = self.block[0](x)
            x = torch.cat((x, skip), dim=1)
            x = self.block[1](x)
            x = x.permute(0, 2, 3, 1)
            B, H, W, C = x.shape
            x = x.view(B, -1, C)
            x = self.block[2](x)
            x = self.block[3](x)
        else:
            x = self.block[0](x)
            x = torch.cat((x, skip), dim=1)
            x = x.permute(0, 2, 3, 1)
            B, H, W, C = x.shape
            x = x.view(B, -1, C)
            x = self.block[1](x)
            x = self.block[2](x)
        return x


class MTUNet(nn.Module):
    def __init__(self, out_ch=4):
        super(MTUNet, self).__init__()
        self.stem = Stem()
        self.encoder = nn.ModuleList()
        self.bottleneck = nn.Sequential(EAmodule(configs["bottleneck"]),
                                        EAmodule(configs["bottleneck"]))
        self.decoder = nn.ModuleList()

        self.decoder_stem = DecoderStem()
        for i in range(len(configs["encoder"])):
            dim = configs["encoder"][i]
            self.encoder.append(encoder_block(dim))
        for i in range(len(configs["decoder"]) - 1):
            dim = configs["decoder"][i]
            self.decoder.append(decoder_block(dim, False))
        self.decoder.append(decoder_block(configs["decoder"][-1], True))
        self.SegmentationHead = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features = self.stem(x)  #(B, N, C) (1, 196, 256)
        skips = []
        for i in range(len(self.encoder)):
            x, skip = self.encoder[i](x)
            skips.append(skip)
            B, C, H, W = x.shape  #  (1, 512, 8, 8)
            x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C)  # (B, N, C)
        x = self.bottleneck(x)  # (1, 25, 1024)
        B, N, C = x.shape
        x = x.view(B, int(np.sqrt(N)), -1, C).permute(0, 3, 1, 2)
        for i in range(len(self.decoder)):
            x = self.decoder[i](x,
                                skips[len(self.decoder) - i - 1])  # (B, N, C)
            B, N, C = x.shape
            x = x.view(B, int(np.sqrt(N)), int(np.sqrt(N)),
                       C).permute(0, 3, 1, 2)

        x = self.decoder_stem(x, features)
        x = self.SegmentationHead(x)
        return x

class MTUNet(nn.Module):
    def __init__(self, out_ch=4):
        super(MTUNet, self).__init__()
        self.stem = Stem()
        self.encoder = nn.ModuleList()
        self.bottleneck = nn.Sequential(EAmodule(configs["bottleneck"]),
                                        EAmodule(configs["bottleneck"]))
        self.decoder = nn.ModuleList()

        self.decoder_stem = DecoderStem()
        for i in range(len(configs["encoder"])):
            dim = configs["encoder"][i]
            self.encoder.append(encoder_block(dim))
        for i in range(len(configs["decoder"]) - 1):
            dim = configs["decoder"][i]
            self.decoder.append(decoder_block(dim, False))
        self.decoder.append(decoder_block(configs["decoder"][-1], True))
        self.SegmentationHead = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features = self.stem(x)  #(B, N, C) (1, 196, 256)
        skips = []
        for i in range(len(self.encoder)):
            x, skip = self.encoder[i](x)
            skips.append(skip)
            B, C, H, W = x.shape  #  (1, 512, 8, 8)
            x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C)  # (B, N, C)
        x = self.bottleneck(x)  # (1, 25, 1024)
        B, N, C = x.shape
        x = x.view(B, int(np.sqrt(N)), -1, C).permute(0, 3, 1, 2)
        for i in range(len(self.decoder)):
            x = self.decoder[i](x,
                                skips[len(self.decoder) - i - 1])  # (B, N, C)
            B, N, C = x.shape
            x = x.view(B, int(np.sqrt(N)), int(np.sqrt(N)),
                       C).permute(0, 3, 1, 2)

        x = self.decoder_stem(x, features)
        x = self.SegmentationHead(x)
        return x
class MTUHSINet02(nn.Module):
    def __init__(self, dataset, num_tokens=4, dim=128, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        super(MTUHSINet02, self).__init__()
        self.dataset = dataset

        channelsNum = [32, 64, 128]

        # self.spe_conv00 = nn.Sequential(
        #     nn.Conv2d(in_channels=get_in_channel(dataset), out_channels=channelsNum[0], kernel_size=(1, 1), stride=(1, 1)),
        #     nn.BatchNorm2d(channelsNum[0]),
        # )
        self.spe_conv3d01 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=channelsNum[0], kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(channelsNum[0]),
            # nn.Dropout3d(p=0.1),
        )
        self.spe_conv3d02 = nn.Sequential(
            nn.Conv3d(in_channels=channelsNum[0], out_channels=channelsNum[1], kernel_size=(5, 3, 3), stride=(1, 1, 1),
                      padding=(2, 1, 1)),
            nn.BatchNorm3d(channelsNum[1]),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.1),

            nn.Conv3d(in_channels=channelsNum[1], out_channels=channelsNum[0], kernel_size=(7, 3, 3), stride=(1, 1, 1),
                      padding=(3, 1, 1)),
            nn.BatchNorm3d(channelsNum[0]),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.1),
        )
        # in_channels=channelsNum[0]*Band15
        self.spa_conv2d01 = nn.Sequential(
            nn.Conv2d(in_channels=channelsNum[0] * 15, out_channels=channelsNum[1], kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0)),
            nn.BatchNorm2d(channelsNum[1]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )
        # Transformer
        self.transformer01 =nn.Sequential(EAmodule(channelsNum[1]),
                                          EAmodule(channelsNum[1]))

        self.spa_conv2d02 = nn.Sequential(
            nn.Conv2d(in_channels=channelsNum[1], out_channels=channelsNum[2], kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.BatchNorm2d(channelsNum[2]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(in_channels=channelsNum[2], out_channels=channelsNum[1], kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.BatchNorm2d(channelsNum[1]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )
        # Transformer
        # self.dim = channelsNum[1]
        self.transformer02 = nn.Sequential(EAmodule(channelsNum[2]),
                                           EAmodule(channelsNum[2]))

        self.spa_conv2d03 = nn.Sequential(
            nn.Conv2d(in_channels=channelsNum[1], out_channels=channelsNum[1], kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0)),
            nn.BatchNorm2d(channelsNum[1]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )

        # # 参考SSRN
        # self.avgpool = nn.AvgPool2d(kernel_size=get_PPsize())
        # self.classifier = nn.Linear(channelsNum[2], get_class_num(dataset))
        # 参考Hybrid
        # get_fc_in(dataset, 'HybridSN')=64 * PPsize * PPsize
        self.FC1 = nn.Sequential(
            nn.Linear(channelsNum[2]*get_PPsize()*get_PPsize(), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )
        self.FC2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )
        self.classifier = nn.Linear(128, out_class)

        self.transformer03 = GWTransformer(dim=128,channelsNum=channelsNum[1])
        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.nn2 = nn.Linear(64, out_class)

        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std=1e-6)

    def forward(self, x, mask=None):

        # local and global spectral feature extraction
        # x=self.spe_conv00(x)
        Spatial_x1 = x
        spectral_x1 = x
        Spatial_x2 = x
        spectral_x2 = x

        spectral_x1 = torch.unsqueeze(spectral_x1, dim=1)  #
        spectral_x1 = self.spe_conv3d01(spectral_x1)
        for iterNum in range(1):
            # print("X1", x.size())
            spe_conv3d = self.spe_conv3d02(spectral_x1)
            spectral_x1 = F.leaky_relu(spe_conv3d + spectral_x1)

        spectral_x1 = rearrange(spectral_x1, 'b c h w y -> b (c h) w y')
        spectral_x1 = self.spa_conv2d01(spectral_x1)

        spectral_x1 = rearrange(spectral_x1, 'b c h w -> b (h w) c')
        trans01 = self.transformer01(spectral_x1, mask)  # main game
        spectral_x1 = F.leaky_relu(trans01 + spectral_x1)
        # trans01 = self.transformer01(x, mask)  # main game
        # x = F.leaky_relu(trans01 + x)
        spectral_x1 = rearrange(spectral_x1, 'b (h w) c -> b c h w', h=get_PPsize(), w=get_PPsize())
        # print("spe_conv3d2", spe_conv3d.size())

        spectral_x2 = torch.unsqueeze(spectral_x2, dim=1)  #
        spectral_x2 = self.spe_conv3d01(spectral_x2)
        for iterNum in range(1):
            # print("X1", x.size())
            spe_conv3d = self.spe_conv3d02(spectral_x2)
            spectral_x2 = F.leaky_relu(spe_conv3d + spectral_x2)

        spectral_x2 = rearrange(spectral_x2, 'b c h w y -> b (c h) w y')
        spectral_x2 = self.spa_conv2d01(spectral_x2)

        spectral_x2 = rearrange(spectral_x2, 'b c h w -> b (h w) c')
        trans01 = self.transformer01(spectral_x2, mask)  # main game
        spectral_x2 = F.leaky_relu(trans01 + spectral_x2)
        # trans01 = self.transformer01(x, mask)  # main game
        # x = F.leaky_relu(trans01 + x)
        spectral_x2 = rearrange(spectral_x2, 'b (h w) c -> b c h w', h=get_PPsize(), w=get_PPsize())
        # print("spe_conv3d2", spe_conv3d.size())

        Spatial_x1 = self.spe_conv00(Spatial_x1)
        # local and global spatial feature extraction
        for iterNum in range(1):
            spa_conv2d = self.spa_conv2d02(Spatial_x1)
            x = F.leaky_relu(spa_conv2d + Spatial_x1)

        x = rearrange(x, 'b c h w -> b (h w) c')
        # print("X1", spa_conv2d.size())
        trans02 = self.transformer02(x, mask)  # main game
        x = F.leaky_relu(trans02 + x)
        # trans02 = self.transformer02(x, mask)  # main game
        # x= F.leaky_relu(trans02  + x)
        # print("X2", spa_conv2d.size())
        Spatial_x1 = rearrange(x, 'b (h w) c -> b c h w', h=get_PPsize(), w=get_PPsize())

        Spatial_x2 = self.spe_conv00(Spatial_x2)
        # local and global spatial feature extraction
        for iterNum in range(1):
            spa_conv2d = self.spa_conv2d02(Spatial_x2)
            x = F.leaky_relu(spa_conv2d + Spatial_x2)

        x = rearrange(x, 'b c h w -> b (h w) c')
        # print("X1", spa_conv2d.size())
        trans02 = self.transformer02(x, mask)  # main game
        x = F.leaky_relu(trans02 + x)
        # trans02 = self.transformer02(x, mask)  # main game
        # x= F.leaky_relu(trans02  + x)
        # print("X2", spa_conv2d.size())
        Spatial_x2 = rearrange(x, 'b (h w) c -> b c h w', h=get_PPsize(), w=get_PPsize())

        # x = Spatial_x1 + Spatial_x2 + spectral_x1 + spectral_x2

        x = torch.cat([Spatial_x1, Spatial_x2, spectral_x1, spectral_x2], dim=1)


        # x = self.spa_conv2d03(x)

        # # 参考SSRN
        # x = self.avgpool(x)
        # x = torch.squeeze(x)
        # x = self.classifier(x)
        #
        # return x

        # 参考Hybrid
        x = torch.reshape(x, (x.shape[0], -1))
        # print(x.size())  #这里打印看下输入全连接层前feature map的大小,self.fc1 = nn.Linear(2000, 500)  # 输入通道数是2000，输出通道数是500
        fc1 = self.FC1(x)
        fc2 = self.FC2(fc1)
        # print(fc2.size())
        out = self.classifier(fc2)
        return out

        # print("X1", x.size())


        # # 参考SSFTT
        # x = rearrange(x, 'b c h w -> b (h w) c')
        # wa = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        # A = torch.einsum('bij,bjk->bik', x, wa)
        # A = rearrange(A, 'b h w -> b w h')  # Transpose
        # A = A.softmax(dim=-1)
        #
        # VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        # T = torch.einsum('bij,bjk->bik', A, VV)
        # cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        #
        # x = torch.cat((cls_tokens, T), dim=1)
        # x += self.pos_embedding
        # x = self.dropout(x)
        # # print("X2", x.size())

        # trans03 = self.transformer03(x)  # main game
        # x = self.to_cls_token(trans03[:, 0])
        # # print("X2", x.size())
        # # import sys
        # # sys.exit(0)
        #
        # x = self.nn1(x)
        # x = self.nn2(x)
        # print("X2", x.size())
        # return x
class MTUHSINet(nn.Module):
    def __init__(self, dataset, num_tokens=4, dim=128, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        super(MTUHSINet, self).__init__()
        self.dataset = dataset

        channelsNum = [32, 64, 128]

        # self.spe_conv00 = nn.Sequential(
        #     nn.Conv2d(in_channels=get_in_channel(dataset), out_channels=channelsNum[0], kernel_size=(1, 1), stride=(1, 1)),
        #     nn.BatchNorm2d(channelsNum[0]),
        # )
        self.spe_conv3d01 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=channelsNum[0], kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(channelsNum[0]),
            # nn.Dropout3d(p=0.1),
        )
        self.spe_conv3d02 = nn.Sequential(
            nn.Conv3d(in_channels=channelsNum[0], out_channels=channelsNum[1], kernel_size=(5, 3, 3), stride=(1, 1, 1),
                      padding=(2, 1, 1)),
            nn.BatchNorm3d(channelsNum[1]),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.1),

            nn.Conv3d(in_channels=channelsNum[1], out_channels=channelsNum[0], kernel_size=(7, 3, 3), stride=(1, 1, 1),
                      padding=(3, 1, 1)),
            nn.BatchNorm3d(channelsNum[0]),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.1),
        )
        # in_channels=channelsNum[0]*Band15
        self.spa_conv2d01 = nn.Sequential(
            nn.Conv2d(in_channels=channelsNum[0] * in_channel, out_channels=channelsNum[1], kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0)),
            nn.BatchNorm2d(channelsNum[1]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )
        # Transformer
        self.transformer01 =nn.Sequential(EAmodule(channelsNum[1]),
                                          EAmodule(channelsNum[1]))

        self.spe_conv00 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=channelsNum[1], kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(channelsNum[1]),
        )

        self.spa_conv2d02 = nn.Sequential(
            nn.Conv2d(in_channels=channelsNum[1], out_channels=channelsNum[2], kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.BatchNorm2d(channelsNum[2]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(in_channels=channelsNum[2], out_channels=channelsNum[1], kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.BatchNorm2d(channelsNum[1]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )
        # Transformer
        # self.dim = channelsNum[1]
        self.transformer02 = nn.Sequential(EAmodule(channelsNum[1]),
                                          EAmodule(channelsNum[1]))

        self.spa_conv2d03 = nn.Sequential(
            nn.Conv2d(in_channels=channelsNum[1]*2, out_channels=channelsNum[1], kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0)),
            nn.BatchNorm2d(channelsNum[1]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )

        # # 参考SSRN
        # self.avgpool = nn.AvgPool2d(kernel_size=get_PPsize())
        # self.classifier = nn.Linear(channelsNum[2], get_class_num(dataset))
        # 参考Hybrid
        # get_fc_in(dataset, 'HybridSN')=64 * PPsize * PPsize
        # self.FC1 = nn.Sequential(
        #     nn.Linear(channelsNum[2]*get_PPsize()*get_PPsize(), 256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.4),
        # )
        # self.FC2 = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.4),
        # )
        # self.classifier = nn.Linear(128, get_class_num(dataset))

        # self.transformer03 = GWTransformer(dim=128,channelsNum=channelsNum[1])
        self.transformer03 = nn.Sequential(EAmodule(channelsNum[1]),
                                           EAmodule(channelsNum[1]))
        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Sequential(
            nn.Linear(channelsNum[1], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.nn2 = nn.Linear(64, out_class)

        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std=1e-6)

    def forward(self, x, mask=None):

        # local and global spectral feature extraction
        # x=self.spe_conv00(x)
        Spatial_x1 = x
        spectral_x1 = x
        Spatial_x2 = x
        spectral_x2 = x

        spectral_x1 = torch.unsqueeze(spectral_x1, dim=1)  #
        spectral_x1 = self.spe_conv3d01(spectral_x1)
        for iterNum in range(2):
            # print("X1", x.size())
            spe_conv3d = self.spe_conv3d02(spectral_x1)
            spectral_x1 = F.leaky_relu(spe_conv3d + spectral_x1)

        spectral_x1 = rearrange(spectral_x1, 'b c h w y -> b (c h) w y')
        spectral_x1 = self.spa_conv2d01(spectral_x1)

        spectral_x1 = rearrange(spectral_x1, 'b c h w -> b (h w) c')
        trans01 = self.transformer01(spectral_x1)  # main game
        spectral_x1 = F.leaky_relu(trans01 + spectral_x1)
        # trans01 = self.transformer01(x, mask)  # main game
        # x = F.leaky_relu(trans01 + x)
        spectral_x1 = rearrange(spectral_x1, 'b (h w) c -> b c h w', h=get_PPsize(), w=get_PPsize())
        # print("spe_conv3d2", spe_conv3d.size())

        # spectral_x2 = torch.unsqueeze(spectral_x2, dim=1)  #
        # spectral_x2 = self.spe_conv3d01(spectral_x2)
        # for iterNum in range(1):
        #     # print("X1", x.size())
        #     spe_conv3d = self.spe_conv3d02(spectral_x2)
        #     spectral_x2 = F.leaky_relu(spe_conv3d + spectral_x2)
        #
        # spectral_x2 = rearrange(spectral_x2, 'b c h w y -> b (c h) w y')
        # spectral_x2 = self.spa_conv2d01(spectral_x2)
        #
        # spectral_x2 = rearrange(spectral_x2, 'b c h w -> b (h w) c')
        # trans01 = self.transformer01(spectral_x2)  # main game
        # spectral_x2 = F.leaky_relu(trans01 + spectral_x2)
        # # trans01 = self.transformer01(x, mask)  # main game
        # # x = F.leaky_relu(trans01 + x)
        # spectral_x2 = rearrange(spectral_x2, 'b (h w) c -> b c h w', h=get_PPsize(), w=get_PPsize())
        # # print("spe_conv3d2", spe_conv3d.size())

        Spatial_x1 = self.spe_conv00(Spatial_x1)
        # local and global spatial feature extraction
        for iterNum in range(1):
            spa_conv2d = self.spa_conv2d02(Spatial_x1)
            x = F.leaky_relu(spa_conv2d + Spatial_x1)

        x = rearrange(x, 'b c h w -> b (h w) c')
        # print("X1", spa_conv2d.size())
        trans02 = self.transformer02(x)  # main game
        x = F.leaky_relu(trans02 + x)
        # trans02 = self.transformer02(x, mask)  # main game
        # x= F.leaky_relu(trans02  + x)
        # print("X2", spa_conv2d.size())
        Spatial_x1 = rearrange(x, 'b (h w) c -> b c h w', h=get_PPsize(), w=get_PPsize())

        # Spatial_x2 = self.spe_conv00(Spatial_x2)
        # # local and global spatial feature extraction
        # for iterNum in range(1):
        #     spa_conv2d = self.spa_conv2d02(Spatial_x2)
        #     x = F.leaky_relu(spa_conv2d + Spatial_x2)
        #
        # x = rearrange(x, 'b c h w -> b (h w) c')
        # # print("X1", spa_conv2d.size())
        # trans02 = self.transformer02(x)  # main game
        # x = F.leaky_relu(trans02 + x)
        # # trans02 = self.transformer02(x, mask)  # main game
        # # x= F.leaky_relu(trans02  + x)
        # # print("X2", spa_conv2d.size())
        # Spatial_x2 = rearrange(x, 'b (h w) c -> b c h w', h=get_PPsize(), w=get_PPsize())

        # x =  F.leaky_relu(Spatial_x1 +  spectral_x1)

        x = torch.cat([Spatial_x1, spectral_x1], dim=1)

        x = self.spa_conv2d03(x)

        # 参考SSRN
        # x = self.avgpool(x)
        # x = torch.squeeze(x)
        # x = self.classifier(x)

        # return x

        # 参考Hybrid
        # x = torch.reshape(x, (x.shape[0], -1))
        # # print(x.size())  #这里打印看下输入全连接层前feature map的大小,self.fc1 = nn.Linear(2000, 500)  # 输入通道数是2000，输出通道数是500
        # fc1 = self.FC1(x)
        # fc2 = self.FC2(fc1)
        # # print(fc2.size())
        # out = self.classifier(fc2)
        # return out

        # print("X1", x.size())


        # # 参考SSFTT
        # x = rearrange(x, 'b c h w -> b (h w) c')
        # wa = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        # A = torch.einsum('bij,bjk->bik', x, wa)
        # A = rearrange(A, 'b h w -> b w h')  # Transpose
        # A = A.softmax(dim=-1)
        #
        # VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        # T = torch.einsum('bij,bjk->bik', A, VV)
        # cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        #
        # x = torch.cat((cls_tokens, T), dim=1)
        # x += self.pos_embedding
        # x = self.dropout(x)
        # # print("X2", x.size())
        # trans03 = self.transformer03(x)  # main game
        # # x = trans03 + x

        x = rearrange(x, 'b c h w -> b (h w) c')
        # print("X1", spa_conv2d.size())
        trans03 = self.transformer03(x)  # main game
        x = F.leaky_relu(trans02 + x)

        x = self.to_cls_token(trans03[:, 0])
        # print("X2", x.size())
        # import sys
        # sys.exit(0)

        x = self.nn1(x)
        x = self.nn2(x)
        # print("X2", x.size())
        return x
configs = {
    "win_size": 4,
    "head": 8,
    "axis": [28, 16, 8],
    "encoder": [256, 512],
    "bottleneck": 1024,
    "decoder": [1024, 512],
    "decoder_stem": [(256, 512), (256, 256), (128, 64), 32]
}
