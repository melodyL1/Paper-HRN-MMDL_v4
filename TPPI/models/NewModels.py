from einops import rearrange
import torch.nn.init as init
from TPPI.models.utils import *
import yaml

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
class MultiModelTrans(nn.Module):

    def __init__(self, dataset, num_tokens=4, dim=128, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        super(MultiModelTrans, self).__init__()

        self.spectral1 = 0
        self.spectral2 = 1
        self.spectral3 = 2
        self.Spatial = 3
        self.dataset = dataset
        self.modes_number = modes_number
        self.num_modalities = len(modes_number)
        self.out_class = out_class
        channelsNum = [32, 64, 128]

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
            nn.Conv2d(in_channels=channelsNum[0]*(self.modes_number[self.spectral1]+self.modes_number[self.spectral2]), out_channels=channelsNum[1], kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0)),
            nn.BatchNorm2d(channelsNum[1]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )
        self.spa_conv2d01b = nn.Sequential(
            nn.Conv2d(in_channels=channelsNum[0]*self.modes_number[self.spectral3], out_channels=channelsNum[1], kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0)),
            nn.BatchNorm2d(channelsNum[1]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )

        # Transformer
        self.transformer01 = Transformer(channelsNum[1],depth, heads, mlp_dim, dropout)
        self.transformer01b = Transformer(channelsNum[1],depth, heads, mlp_dim, dropout)

        self.FE = nn.Sequential(
            nn.Conv2d(in_channels=self.modes_number[self.Spatial], out_channels=channelsNum[1], kernel_size=(1, 1), stride=(1, 1)),
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
            nn.Conv2d(in_channels=channelsNum[1]*3, out_channels=channelsNum[2], kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.BatchNorm2d(channelsNum[2]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )

        # Tokenization
        self.transformer03 = GWTransformer(channelsNum=channelsNum[2])

        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.nn2 = nn.Linear(64, out_class)

        # line
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std=1e-6)


    def forward(self, x, mask=None):

        # 按模态分割输入并分别处理
        start = 0
        modality_input = []
        for i, channels in enumerate(self.modes_number):
            end = start + channels
            modality_input.append(x[:, start:end, :, :])
            start = end
        spectral_x1 = modality_input[self.spectral1]
        spectral_x2 = modality_input[self.spectral2]
        spectral_x3 = modality_input[self.spectral2]
        Spatial_x1 = modality_input[self.Spatial]

        # 3DCNN + transformer
        spectral_x = torch.cat([spectral_x1, spectral_x2], dim=1)
        spectral_x = torch.unsqueeze(spectral_x, dim=1)
        spectral_x = self.spe_conv3d01(spectral_x)

        for iterNum in range(2):
            spe_conv3d = self.spe_conv3d02(spectral_x)
            spectral_x = spe_conv3d + spectral_x

        spectral_x = rearrange(spectral_x, 'b c h w y -> b (c h) w y')
        # 通道数 = 32*modality_input[spectral]的通道数
        spectral_x = self.spa_conv2d01(spectral_x)

        trans01 = rearrange(spectral_x, 'b c h w -> b (h w) c')
        trans01 = self.transformer01(trans01)  # main game
        trans01 = rearrange(trans01, 'b (h w) c -> b c h w', h=get_PPsize(), w=get_PPsize())
        spectral_x = trans01  + spectral_x

        # 3DCNN + transformer(b)
        spectral_xb = spectral_x3
        spectral_xb = torch.unsqueeze(spectral_xb, dim=1)
        spectral_xb = self.spe_conv3d01b(spectral_xb)

        for iterNum in range(2):
            spe_conv3db = self.spe_conv3d02b(spectral_xb)
            spectral_xb = spe_conv3db + spectral_xb

        spectral_xb = rearrange(spectral_xb, 'b c h w y -> b (c h) w y')
        # 通道数 = 32*modality_input[spectral]的通道数
        spectral_xb = self.spa_conv2d01b(spectral_xb)

        trans01b = rearrange(spectral_xb, 'b c h w -> b (h w) c')
        trans01b = self.transformer01b(trans01b)  # main game
        trans01b = rearrange(trans01b, 'b (h w) c -> b c h w', h=get_PPsize(), w=get_PPsize())
        spectral_xb = trans01b  + spectral_xb

        # 2DCNN + transformer
        Spatial_x = Spatial_x1
        Spatial_x = self.FE(Spatial_x)
        for iterNum in range(2):
            spa_conv2d= self.spa_conv2d02(Spatial_x)
            Spatial_x = spa_conv2d + Spatial_x

        trans02 = rearrange(Spatial_x, 'b c h w -> b (h w) c')
        trans02 = self.transformer02(trans02)  # main game
        trans02 = rearrange(trans02, 'b (h w) c -> b c h w',h=get_PPsize(), w=get_PPsize())
        Spatial_x = trans02 + Spatial_x

        x = torch.cat([spectral_x, spectral_xb, Spatial_x], dim=1)
        # 通道数 = channelsNum[1]*cat函数拼接个数
        x = self.spa_conv2d03(x)

        trans03 = self.transformer03(x)

        x = self.to_cls_token(trans03[:, 0])

        x = self.nn1(x)
        x = self.nn2(x)

        return x