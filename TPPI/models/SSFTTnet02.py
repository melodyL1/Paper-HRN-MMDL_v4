
from einops import rearrange
from torch import nn
import torch.nn.init as init
from TPPI.models.utils import *



def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# 等于 FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
# 等于 FeedForward
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

NUM_CLASS = 20

class SSFTTnet(nn.Module):
    def __init__(self, dataset, in_channels=1, num_classes=NUM_CLASS, num_tokens=4, dim=64, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        super(SSFTTnet, self).__init__()
        self.dataset = dataset
        self.L = num_tokens
        self.cT = dim
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
        #Xiongan in_channels=8*13, IP in_channels=8*28
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8*13, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Tokenization
        self.token_wA = nn.Parameter(torch.empty(1, self.L, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

    def forward(self, x, mask=None):
        # print(x.size())
        x = torch.unsqueeze(x, dim=1)
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        # print(x.size())
        x = self.conv2d_features(x)
        x = rearrange(x,'b c h w -> b (h w) c')

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
        x = self.transformer(x, mask)  # main game
        x = self.to_cls_token(x[:, 0])
        x = self.nn1(x)

        return x
# @torchsnooper.snoop()
class SSRNTransOLD(nn.Module):
    """
    Based on paper:Zhong, Z. Spectral-Spatial Residual Network for Hyperspectral Image Classification: A 3-D Deep Learning Framework. TGRS
    Input shape:[N,C=spectral_channel,H=5,W=5]
    """
    def __init__(self, dataset, num_tokens=4, dim=64, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        super(SSRNTrans, self).__init__()
        self.dataset = dataset

        channelsNum = 64

        self.FE1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=channelsNum, kernel_size=(7, 1, 1), stride=(2, 1, 1)),
            nn.BatchNorm3d(channelsNum),
        )
        self.spe_conv1 = nn.Sequential(
            nn.Conv3d(in_channels=channelsNum, out_channels=channelsNum, kernel_size=(7, 1, 1), stride=(1, 1, 1),
                      padding=(3, 0, 0)),
            nn.BatchNorm3d(channelsNum),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=channelsNum, out_channels=channelsNum, kernel_size=(7, 1, 1), stride=(1, 1, 1),
                      padding=(3, 0, 0)),
            nn.BatchNorm3d(channelsNum),
            nn.ReLU(inplace=True),
        )
        self.spe_conv2 = nn.Sequential(
            nn.Conv3d(in_channels=channelsNum, out_channels=channelsNum, kernel_size=(7, 1, 1), stride=(1, 1, 1),
                      padding=(3, 0, 0)),
            nn.BatchNorm3d(channelsNum),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=channelsNum, out_channels=channelsNum, kernel_size=(7, 1, 1), stride=(1, 1, 1),
                      padding=(3, 0, 0)),
            nn.BatchNorm3d(channelsNum),
            nn.ReLU(inplace=True),
        )
        self.CF = nn.Sequential(
            nn.Conv3d(in_channels=channelsNum, out_channels=128, kernel_size=(get_SSRN_channel(dataset), 1, 1),
                      stride=(1, 1, 1)),
            nn.BatchNorm3d(128),
        )
        self.FE2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=channelsNum, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(channelsNum),
        )  # 5*5*128->3*3*24
        self.spa_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channelsNum, out_channels=channelsNum, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.BatchNorm2d(channelsNum),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(in_channels=channelsNum, out_channels=channelsNum, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.BatchNorm2d(channelsNum),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
        )

        self.avgpool = nn.AvgPool2d(kernel_size=get_PPsize()-2)
        self.classifier = nn.Linear(channelsNum, get_class_num(dataset))


        # Tokenization
        self.L = num_tokens
        self.cT = dim
        # 64--24
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

        self.nn1 = nn.Linear(dim, get_class_num(dataset))
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)


    def forward(self, x, mask=None):
        x = torch.unsqueeze(x, dim=1)
        # print("1x)", x.size())
        FE1 = self.FE1(x)
        # print("2X", x.size())
        spe_conv1 = self.spe_conv1(FE1)
        spe_conv1_new = spe_conv1 + FE1
        spe_conv2 = self.spe_conv2(spe_conv1_new)
        spe_conv2_new = spe_conv2 + spe_conv1_new
        # print("spe_conv2_new ", spe_conv2_new .size())
        CF = self.CF(spe_conv2_new)
        # print("CF1 ", CF.size())
        CF = torch.squeeze(CF, dim=2)
        # print("CF2 ", CF.size())
        FE2 = self.FE2(CF)
        # print("FE2 ", FE2.size())

        spa_conv1 = self.spa_conv1(FE2)
        spa_conv1_new = spa_conv1 + FE2
        # spa_conv2 = self.spa_conv2(spa_conv1_new)
        # spa_conv2_new = spa_conv2 + spa_conv1_new


        # avg = self.avgpool(spa_conv1_new)
        # avg = torch.squeeze(avg)
        # out = self.classifier(avg)
        # return out
        # print("spa_conv1_new", spa_conv1_new.size())
        x = rearrange(spa_conv1_new,'b c h w -> b (h w) c')
        # print("rearrange(x)", x.size())

        wa = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        # print("rearrange(self.token_wA)", wa.size())

        A = torch.einsum('bij,bjk->bik', x, wa)
        # import sys
        # sys.exit(0)

        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)

        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x, mask)  # main game
        x = self.to_cls_token(x[:, 0])
        x = self.nn1(x)
        return x
class SSRNTransNEWMobil(nn.Module):
    """
    Based on paper:Zhong, Z. Spectral-Spatial Residual Network for Hyperspectral Image Classification: A 3-D Deep Learning Framework. TGRS
    Input shape:[N,C=spectral_channel,H=5,W=5]
    """
    def __init__(self, dataset, num_tokens=4, dim=128, depth=2, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        super(SSRNTransNEWMobil, self).__init__()
        self.dataset = dataset

        channelsNum = [32, 64, 128]

        # self.spe_conv00 = nn.Sequential(
        #     nn.Conv2d(in_channels=get_in_channel(dataset), out_channels=channelsNum[0], kernel_size=(1, 1), stride=(1, 1)),
        #     nn.BatchNorm2d(channelsNum[0]),
        # )
        self.spe_conv3d01= nn.Sequential(
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
            nn.Conv2d(in_channels=channelsNum[0]*15, out_channels=channelsNum[1], kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0)),
            nn.BatchNorm2d(channelsNum[1]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )
        # Transformer
        self.transformer01 = Transformer(channelsNum[1], depth, heads, mlp_dim, dropout)

        self.spe_conv00 = nn.Sequential(
            nn.Conv2d(in_channels=get_in_channel(dataset), out_channels=channelsNum[1], kernel_size=(1, 1), stride=(1, 1)),
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
        self.transformer02 = Transformer(channelsNum[1], depth, heads, mlp_dim, dropout)

        self.spa_conv2d03 = nn.Sequential(
            nn.Conv2d(in_channels=channelsNum[1]*2+15, out_channels=channelsNum[2], kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.BatchNorm2d(channelsNum[2]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )

        # 参考SSRN
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



        # Tokenization
        self.L = num_tokens
        self.cT = dim

        self.token_wA = nn.Parameter(torch.empty(1, self.L, channelsNum[2] ),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, channelsNum[2] , self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer03 = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()


        self.nn1 = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.nn2 = nn.Linear(64, get_class_num(dataset))

        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std=1e-6)


    def forward(self, x, mask=None):

        # local and global spectral feature extraction
        # x=self.spe_conv00(x)
        Spatial_x1 = x
        spectral_x1= x
        Spatial_x2 = x
        spectral_x2= x

        spectral_x1 = torch.unsqueeze(spectral_x1, dim=1)  #
        spectral_x1 = self.spe_conv3d01(spectral_x1)
        for iterNum in range(2):
            # print("X1", x.size())
            spe_conv3d = self.spe_conv3d02(spectral_x1)
            spectral_x1 = F.leaky_relu(spe_conv3d + spectral_x1)

        spectral_x1 = rearrange(spectral_x1, 'b c h w y -> b (c h) w y')
        spectral_x1 = self.spa_conv2d01(spectral_x1)

        spectral_x1 = rearrange(spectral_x1, 'b c h w -> b (h w) c')
        trans01  = self.transformer01(spectral_x1, mask)  # main game
        spectral_x1 = F.leaky_relu(trans01 + spectral_x1 )
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
        # trans01  = self.transformer01(spectral_x2, mask)  # main game
        # spectral_x2 = F.leaky_relu(trans01 + spectral_x2 )
        # # trans01 = self.transformer01(x, mask)  # main game
        # # x = F.leaky_relu(trans01 + x)
        # spectral_x2 = rearrange(spectral_x2, 'b (h w) c -> b c h w', h=get_PPsize(), w=get_PPsize())
        # # print("spe_conv3d2", spe_conv3d.size())


        Spatial_x1 = self.spe_conv00(Spatial_x1)
        # local and global spatial feature extraction
        for iterNum in range(2):
            spa_conv2d= self.spa_conv2d02(Spatial_x1)
            x = F.leaky_relu(spa_conv2d + Spatial_x1)

        x = rearrange(x,'b c h w -> b (h w) c')
        # print("X1", spa_conv2d.size())
        trans02 = self.transformer02(x, mask)  # main game
        x= F.leaky_relu(trans02  + x)
        # trans02 = self.transformer02(x, mask)  # main game
        # x= F.leaky_relu(trans02  + x)
        # print("X2", spa_conv2d.size())
        Spatial_x1 = rearrange(x, 'b (h w) c -> b c h w',h=get_PPsize(), w=get_PPsize())


        # Spatial_x2 = self.spe_conv00(Spatial_x2)
        # # local and global spatial feature extraction
        # for iterNum in range(1):
        #     spa_conv2d= self.spa_conv2d02(Spatial_x2)
        #     x = F.leaky_relu(spa_conv2d + Spatial_x2)
        #
        # x = rearrange(x,'b c h w -> b (h w) c')
        # # print("X1", spa_conv2d.size())
        # trans02 = self.transformer02(x, mask)  # main game
        # x= F.leaky_relu(trans02  + x)
        # # trans02 = self.transformer02(x, mask)  # main game
        # # x= F.leaky_relu(trans02  + x)
        # # print("X2", spa_conv2d.size())
        # Spatial_x2 = rearrange(x, 'b (h w) c -> b c h w',h=get_PPsize(), w=get_PPsize())

        # x = Spatial_x1 + spectral_x1 + Spatial_x2

        x =torch.cat([Spatial_x1,spectral_x1,Spatial_x2],dim = 1)

        x = self.spa_conv2d03(x)

        # 参考SSRN
        # x = self.avgpool(x)
        # x = torch.squeeze(x)
        # x = self.classifier(x)
        # 参考Hybrid
        # x = torch.reshape(x, (x.shape[0], -1))
        # # print(x.size())  #这里打印看下输入全连接层前feature map的大小,self.fc1 = nn.Linear(2000, 500)  # 输入通道数是2000，输出通道数是500
        # fc1 = self.FC1(x)
        # fc2 = self.FC2(fc1)
        # # print(fc2.size())
        # out = self.classifier(fc2)
        # return out

        # # print("X1", x.size())
        # 参考SSFTT
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
        # print("X2", x.size())
        trans03 = self.transformer03(x, mask)  # main game
        x =F.relu( trans03 + x )
        # trans03 = self.transformer03(x, mask)  # main game
        # x =F.relu( trans03 + x )
        # print("X2", x.size())
        x = self.to_cls_token(x[:, 0])
        # print("X2", x.size())
        # import sys
        # sys.exit(0)

        x = self.nn1(x)
        x = self.nn2(x)
        return x
