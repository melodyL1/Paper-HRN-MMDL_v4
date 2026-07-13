from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from TPPI.models.utils import *
import yaml
from fightingcv_attention.attention.SEAttention import SEAttention

try:
    from mamba_ssm import Mamba
except ImportError:
    print("Warning: mamba_ssm not installed. Please install it to use Mamba layer.")
    Mamba = None


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

# 加载参数
with open("configs/config.yml") as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)
    modes_number = string_to_int_list(cfg['Data']['modes_number'])
    started = cfg["Data"]["band_selection"][0]
    end = cfg["Data"]["band_selection"][1]
    in_channel = end - started
    out_class = cfg['Data']['class']
    PPsize = cfg['Preprocessing']['PP_size']

# 模块
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

class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.nn1(out)
        out = self.do1(out)
        return out

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
            x = attention(x, mask=mask)
            x = mlp(x)
        return x

class GWTransformer(nn.Module):

    def __init__(self, num_tokens=4, dim=128, channelsNum=64, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        super(GWTransformer, self).__init__()

        self.L = num_tokens
        self.cT = dim

        self.token_wA = nn.Parameter(torch.empty(1, self.L, channelsNum),requires_grad=True)
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, channelsNum, self.cT),requires_grad=True)
        torch.nn.init.xavier_normal_(self.token_wV)
        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()


    def forward(self, x):

        x = rearrange(x, 'b c h w -> b (h w) c')
        wa = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
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

# 模型
class NewModels(nn.Module):

    def __init__(self, in_channels=18, out_class=7, num_tokens=4, dim=128, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1,split_n=6):
        super(NewModels, self).__init__()
        self.split_n=split_n

        channelsNum = [32, 64, 128]
        self.out_class = out_class

        #self.modes_number = [in_channels]
        #self.Spatial1 = 0  # 现在的单模态就相当于原来的 Spatial 模态
        #self.Spatial2= 0
        #self.Spatial = 0

        self.spe_conv3d01= nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=channelsNum[0], kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(channelsNum[0]),
        )
        self.spe_conv3d01b= nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=channelsNum[0], kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(channelsNum[0]),
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
        self.spe_conv3d02b = nn.Sequential(
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

        self.spa_conv2d01 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=channelsNum[1], kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0)),
            nn.BatchNorm2d(channelsNum[1]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )
        self.spa_conv2d01b = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=channelsNum[1], kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0)),
            nn.BatchNorm2d(channelsNum[1]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )

        # Transformer
        self.transformer01 = Transformer(channelsNum[1],depth, heads, mlp_dim, dropout)
        self.transformer01b = Transformer(channelsNum[1],depth, heads, mlp_dim, dropout)

        self.FE = nn.Sequential(
            nn.Conv2d(in_channels=split_n, out_channels=channelsNum[1], kernel_size=(1, 1), stride=(1, 1)),
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
        self.transformer02 = Transformer(channelsNum[1],depth, heads, mlp_dim, dropout)

        self.spa_conv2d03 = nn.Sequential(
            nn.Conv2d(in_channels=channelsNum[1], out_channels=channelsNum[2], kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.BatchNorm2d(channelsNum[2]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )

        # Tokenization
        self.transformer03 = GWTransformer(channelsNum=channelsNum[2])

        self.to_cls_token = nn.Identity()

        dim_upper = dim

        dim_lower = channelsNum[0]

        fusion_in_features = dim_upper + dim_lower

        self.se_attention = SEAttention(channel=fusion_in_features,reduction=8)

        if Mamba is not None:
            self.mamba_upper = Mamba(
                d_model = dim_upper,
                d_state=16,
                d_conv=4,
                expand = 2
            )

            self.mamba_lower = Mamba(
                d_model = dim_lower,
                d_state=16,
                d_conv=4,
                expand = 2
            )

            self.mamba_fusion = Mamba(
                d_model = fusion_in_features,
                d_state=16,
                d_conv=4,
                expand = 2
            )

        else:
            self.mamba_upper = nn.Identity()
            self.mamba_lower = nn.Identity()
            self.mamba_fusion = nn.Identity()





        self.nn1 = nn.Sequential(
            nn.Linear(fusion_in_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.nn2 = nn.Linear(64, out_class)

        # line
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std=1e-6)


    def forward(self, x, mask=None):

        x_upper = x[:,:self.split_n,:,:]
        x_lower = x[:,self.split_n:,:,:]

        feat_upper = self.FE(x_upper)
        for iterNum in range(2):
            spa_conv2d= self.spa_conv2d02(feat_upper)
            feat_upper = spa_conv2d + feat_upper

        b,c,h,w = feat_upper.shape
        trans02 = rearrange(feat_upper, 'b c h w -> b (h w) c')
        trans02 = self.transformer02(trans02)  # main game
        trans02 = rearrange(trans02, 'b (h w) c -> b c h w', h=h, w=w)
        feat_upper = trans02 + feat_upper


        # 通道数 = channelsNum[1]*cat函数拼接个数
        feat_upper = self.spa_conv2d03(feat_upper)

        trans03 = self.transformer03(feat_upper)

        token_upper = self.to_cls_token(trans03[:, 0])

        token_upper = token_upper.unsqueeze(1)
        token_upper = self.mamba_upper(token_upper)
        token_upper = token_upper.squeeze(1)



        x_lower = x_lower.unsqueeze(1)
        feat_lower=self.spe_conv3d01(x_lower)
        feat_lower=self.spe_conv3d02(feat_lower)

        feat_lower=F.adaptive_avg_pool3d(feat_lower,(1,1,1))
        token_lower=feat_lower.view(feat_lower.size(0), -1)

        token_lower = token_lower.unsqueeze(1)
        token_lower = self.mamba_lower(token_lower)
        token_lower = token_lower.squeeze(1)


        combined=torch.cat((token_upper,token_lower),dim=1)

        combined_4d = combined.unsqueeze(2).unsqueeze(3)
        attended_feat = self.se_attention(combined_4d)

        combined_final = attended_feat.view(attended_feat.size(0), -1)

        mamba_input = combined_final.unsqueeze(1)

        mamba_out = self.mamba_fusion(mamba_input)

        combined_final = mamba_out.squeeze(1)





        out = self.nn1(combined_final)
        out=self.nn2(out)

        return out