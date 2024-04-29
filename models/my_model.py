import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, DropPath, trunc_normal_
import math
from einops import rearrange, repeat
from torchvision.ops import DeformConv2d
from utils.utils import changeshape, changeshape3


# 细节增强
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = torch.mean(x, dim=1, keepdim=True)
        x1 = self.conv1(x1)
        return self.sigmoid(x1)*x
class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=False),
                nn.ReLU(inplace=False),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        x = x * y
        return x
class multi_ca(nn.Module):
    def __init__(self, dim, bins = [1,2,3,6]):
        super(multi_ca, self).__init__()

        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AvgPool2d(bin),
                nn.Conv2d(dim, dim, kernel_size=1),
                CALayer (dim),
            ))
        self.features = nn.ModuleList(self.features)
        self.cov2 = nn.Conv2d(dim * 5, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return self.cov2(torch.cat(out, 1))

class MSAFF(nn.Module):
    def __init__(self, dim):
        super(MSAFF, self).__init__()
        self.multi_ca = multi_ca(dim)

        self.SA1 = SpatialAttention(1)
        self.SA3 = SpatialAttention(3)
        self.SA5 = SpatialAttention(5)
        self.SA7 = SpatialAttention(7)
        self.cov3 = nn.Conv2d(dim*4, dim, kernel_size=3, padding=1)
        self.t = torch.nn.Tanh()
    def forward(self, x):
        x = changeshape(x)
        input_x = x
        x = self.multi_ca(x)

        xx1 = self.SA1(x)
        xx2 = self.SA3(x)
        xx3 = self.SA5(x)
        xx4 = self.SA7(x)
        x = torch.cat((xx1, xx2, xx3, xx4), dim=1)
        x =  self.cov3(x)
        x = self.t(x)+input_x
        x = changeshape3(x)
        return x
#特征融合
class CFF(nn.Module):
    def __init__(self, channel, reduction=8, bias=False):
        super(CFF, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.t = nn.Sequential(
            nn.Conv2d(channel*2, channel//reduction, kernel_size=to_2tuple(3), padding=1, bias=bias),
            # True表示直接修改原始张量
            nn.ReLU(inplace=False),
            nn.Conv2d(channel//reduction, channel*2, kernel_size=to_2tuple(3), padding=1,bias=bias),
            nn.Sigmoid()
        )
    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape
        in_feats = torch.cat(in_feats, dim=1)
        x = in_feats
        pa = self.t(x)
        j = torch.mul(pa, in_feats)
        
        return j
    
class MFF(nn.Module):
    def __init__(self, channel, reduction=8, bias=False):
        super(MFF, self).__init__()

        
        # 自适应平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        if channel == 48:
            self.conv2 = nn.Sequential(nn.Conv2d(channel // 2, channel, kernel_size=3, stride=2, padding=1),
                                       nn.ReLU(inplace=False))
            self.conv1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0),
                                       nn.ReLU(inplace=False))
            self.conv3 = nn.Sequential(nn.ConvTranspose2d(channel*2, channel, kernel_size=3, stride=2, padding=1,output_padding=1),
                                       nn.ReLU(inplace=False))



        elif channel == 24:
            self.conv1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0),
                                       nn.ReLU(inplace=False))
            self.conv2 = nn.Sequential(nn.ConvTranspose2d(channel * 2, channel, kernel_size=3, stride=2, padding=1,output_padding=1),
                                       nn.ReLU(inplace=False))
            self.conv3 = nn.Sequential(nn.ConvTranspose2d(channel * 4, channel, kernel_size=5, stride=4, padding=1,output_padding=1),
                                       nn.ReLU(inplace=False))


        else:
            self.conv2 = nn.Sequential(nn.Conv2d(channel // 4, channel, kernel_size=5, stride=4, padding=1),
                                       nn.ReLU(inplace=False))
            self.conv3 = nn.Sequential(nn.Conv2d(channel // 2, channel, kernel_size=3, stride=2, padding=1),
                                       nn.ReLU(inplace=False))
            self.conv1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0),
                                       nn.ReLU(inplace=False))

        
        self.abc = CFF(channel)
    # 计算X(fused)=X(l,fused)+X(l-1,fused)
    def forward(self, in_feats):
        in_feats[0] = changeshape(in_feats[0])
        in_feats[1] = changeshape(in_feats[1])
        in_feats[2] = changeshape(in_feats[2])
        in_feats[3] = changeshape(in_feats[3])
        in_feats[1] = self.conv1(in_feats[1])
        in_feats[2] = self.conv2(in_feats[2])
        in_feats[3] = self.conv3(in_feats[3])

        f1 = self.abc([in_feats[0], in_feats[1]])
        f2 = self.abc([in_feats[2], in_feats[3]])
        f = f1+f2
        # f = torch.cat([f1,f2],dim=1)
        j = changeshape3(f)
        return j

#ADTB
class TRANSFORMER_BLOCK(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear',token_mlp='ffn',se_layer=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            Deformable_Attentive_Transformer(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, win_size=win_size,
                                 shift_size=0 if (i % 2 == 0) else win_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
            for i in range(depth)])

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"    

    def forward(self, x, mask=None):
        for blk in self.blocks:
            # 判断是否使用checkpoint技术
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x,mask)
        return x
    # 计算计算量
    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops

class Deformable_Attentive_Transformer(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.ReLU, norm_layer=nn.LayerNorm,token_projection='conv',token_mlp='leff',se_layer=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection,se_layer=se_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,act_layer=act_layer, drop=drop) if token_mlp=='ffn' else LeFF(dim,mlp_hidden_dim,act_layer=act_layer, drop=drop)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
    # 注意这儿的mask默认为None
    def forward(self, x, mask=None):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))

        ## input mask
        #这段代码处理了输入的遮罩，在自注意力机制中用于屏蔽一些位置，以防止模型关注到不应该考虑的区域
        if mask != None:
            input_mask = F.interpolate(mask, size=(H,W)).permute(0,2,3,1)

            input_mask_windows = window_partition(input_mask, self.win_size) # nW, win_size, win_size, 1

            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size

            attn_mask = attn_mask.unsqueeze(2)*attn_mask.unsqueeze(1) # nW, win_size*win_size, win_size*win_size

            # 将不为0的位置置为较大的负数，零位置保持为零
            attn_mask = attn_mask.masked_fill(attn_mask!=0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        else:
            attn_mask = None

        ## shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2) # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(shift_attn_mask == 0, float(0.0))
            # print(attn_mask.shape, shift_attn_mask.shape)
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask
            
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C) 
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, win_size*win_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        # 恢复窗口的大小
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        del attn_mask
        return x
class WindowAttention(nn.Module):
    def __init__(self, dim, win_size,num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,se_layer=False):

        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0]) # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1]) # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 可用于消融实验，使用不同的方法获取q,k,v
        if token_projection =='conv':
            self.qkv = ConvProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        elif token_projection =='linear_concat':
            self.qkv = LinearProjection_Concat_kv(dim,num_heads,dim//num_heads,bias=qkv_bias)
        else:
            self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        self.conv = nn.Conv2d(dim, dim, kernel_size=to_2tuple(3), padding=1, groups=dim, bias=False)
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.se_layer = SELayer(dim) if se_layer else nn.Identity()
        # self.se_layer = SELayer(dim) if se_layer else CALayer(dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x,attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))* self.alpha
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1)//relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)
        
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N*ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N*ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)  
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.se_layer(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'
#规范化的多层感知机
# 激活函数改为LeackReLu,去掉Dropout(drop)
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate !=1:
        x = x.permute(0,3,1,2) # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=win_size) # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
    return windows

def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate !=1:
        x = windows.permute(0,5,3,4,1,2).contiguous() # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1),stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU,drop = 0.):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = hh)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h = hh, w = hh)

        x = self.linear2(x)

        return x

# Squeeze-and-Excitation（SE）注意力机制
# 该机制通过学习每个通道的权重，使网络能够自适应地关注输入特征中的重要通道，提高网络对有关信息的表达能力
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 通过两个线性层和一个ReLU激活函数，以及一个Sigmoid函数，学习生成每个通道的注意力权重
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, N, C]
        x = torch.transpose(x, 1, 2)  # [B, C, N]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        x = torch.transpose(x, 1, 2)  # [B, N, C]
        return x

######## Embedding for q,k,v ########
class ConvProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False,bias=True):

        super().__init__()

        inner_dim = dim_head *  heads
        self.heads = heads
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)
        
    def forward(self, x, attn_kv=None):
        # 批次大小('b')，序列长度('n')，通道数('c')，注意力头数（'h'）
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))
        # 检查是否提供了键值对作为输入，没有则将x作为键值对 
        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)
        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        # 根据head的数量，对q k v进行均分
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q,k,v    
    
    def flops(self, H, W): 
        flops = 0
        flops += self.to_q.flops(H, W)
        flops += self.to_k.flops(H, W)
        flops += self.to_v.flops(H, W)
        return flops

class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        attn_kv = x if attn_kv is None else attn_kv
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1] 
        return q,k,v

    def flops(self, H, W): 
        flops = H*W*self.dim*self.inner_dim*3
        return flops 

class LinearProjection_Concat_kv(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        attn_kv = x if attn_kv is None else attn_kv
        qkv_dec = self.to_qkv(x).reshape(B_, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv_enc = self.to_kv(attn_kv).reshape(B_, N, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k_d, v_d = qkv_dec[0], qkv_dec[1], qkv_dec[2]  # make torchscript happy (cannot use tensor as tuple)
        k_e, v_e = kv_enc[0], kv_enc[1] 
        k = torch.cat((k_d,k_e),dim=2)
        v = torch.cat((v_d,v_e),dim=2)
        return q,k,v

    def flops(self, H, W): 
        flops = H*W*self.dim*self.inner_dim*5
        return flops

#用于分离出Δn和Δm
class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,act_layer=nn.ReLU): 
        super(SepConv2d, self).__init__()
        
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # 用于选择激活函数的类型，nn.Identity是一个恒等函数，输入即等于输出
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.offset_conv1 = nn.Conv2d(in_channels, 216, 3, stride=1, padding=1, bias= False)
        self.deform1 =  DeformConv2d(in_channels, out_channels, 3, padding=1, groups=8)
        # self.deform1 =  nn.Sequential(                              
             
        #     nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, groups=8),
        #     nn.Conv2d(in_channels, out_channels, kernel_size=to_2tuple(1), bias=False) 
                      
        # )
        
        self.SA = SpatialAttention()
        # self.PA = PALayer(216)
       
    def offset_gen(self, x):
        sa = self.SA(x) 
        # sa = self.PA(x)
        o1, o2, mask = torch.chunk(sa, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return offset,mask

    def forward(self, x):
        offset1,mask = self.offset_gen(self.offset_conv1(x))
        # 将使用双重并行注意力机制获得的offset1和mask引入可变形卷积中
        feat1 = self.deform1(x, offset1, mask)
        # feat1 = self.deform1(x)           
        x = self.act_layer(feat1)
        x = self.pointwise(x) 
          
        return x

#Spatially Attentive Offset Extractor部分的

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 这儿只有通道维度的最大和平均池化
        # x [1024, 216, 8, 8]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # avg_out [1024, 1, 8, 8]
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # max_out [1024, 1, 8, 8]
        x1 = torch.cat([avg_out, max_out], dim=1)
        # x1 [1024, 2, 8, 8]
        x1 = self.conv1(x1)
        # x1 [1024, 1, 8, 8]
        # 权重函数后面往往跟一个sigmoid函数
        return self.sigmoid(x1)*x


# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None,act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1,2).contiguous()  # B H*W C
        return out

class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel
        
    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1,2).contiguous() # B H*W C
        return out

class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

class Pixel_Shuffle(nn.Module):
    def __init__(self, dim, up_scale=2, bias=False):
        super(Pixel_Shuffle, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
        self.up = nn.PixelShuffle(up_scale)
        self.qk_pre = nn.Conv2d(int(dim // (up_scale ** 2)), 3, kernel_size=to_2tuple(1), bias=bias)
    def forward(self, x):
        x = changeshape(x)
        qk = self.qk_pre(self.up(x))
        fake_image = qk
        x = changeshape3(x)
        return x, fake_image

class Model(nn.Module):
    def __init__(self, in_channel=3, out_channel=4, dim=24, window_size=8,
                #  depths=[8, 8, 8, 8, 4, 4, 4], 
                 #  num_heads=[2, 4, 8, 16, 8, 4, 2],
                 depths=[2, 2, 2, 2, 2, 2, 2], 
                 num_heads=[8, 8, 8, 8, 4, 4, 4],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='conv', token_mlp='ffn', se_layer=False, img_size=256,**kwargs): 
        super(Model, self).__init__()
        # 编码层的层数
        self.num_enc_layers = len(depths)//2
        # 解码层的层数
        self.num_dec_layers = len(depths)//2
        self.embed_dim = dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        # 用于对图像块进行MLP处理的方法，可以选择ffn或linear，这儿默认为ffn
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size =win_size
        self.reso = 256
        # nn.Dropout在PyTorch中用于实现dropout操作的模块，
        # 它可以在训练过程中以一定的概率随机将输入张量某些元素置零，从而防止过拟合
        # 这儿参数p设置为0，相当于禁用了Dropout操作
        self.pos_drop = nn.Dropout(p=drop_rate)
        # 编-解码器中dropout的比率
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))] 
        conv_dpr = [drop_path_rate]*depths[4]
        # 解码器的dropout比率列表，enc_dpr列表的逆序，即将其反转。这个列表用于设置解码器各层的dropout比率
        dec_dpr = enc_dpr[::-1]
        
        self.input_proj = InputProj(in_channel=3, out_channel=dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=3*dim, out_channel=in_channel, kernel_size=3, stride=1)
        self.output_proj1 = OutputProj(in_channel=dim, out_channel=in_channel, kernel_size=3, stride=1)
        
        # Encoder
        self.encoderlayer_0 = TRANSFORMER_BLOCK(dim=dim,
                            input_resolution=(img_size,
                                                img_size),
                            depth=depths[0],
                            num_heads=num_heads[0],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            # drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                            drop_path=0.,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.msaff0 = MSAFF(dim=dim)
        self.dowsample_0 = Downsample(dim, dim*2)
        self.encoderlayer_1 = TRANSFORMER_BLOCK(dim=dim*2,
                            input_resolution=(img_size // 2,
                                                img_size // 2),
                            depth=depths[1],
                            num_heads=num_heads[1],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            # drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                            drop_path=0.,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.msaff1 = MSAFF(dim=dim*2)
        self.dowsample_1 = Downsample(dim*2, dim*4)
        self.encoderlayer_2 = TRANSFORMER_BLOCK(dim=dim*4,
                            input_resolution=(img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                            depth=depths[2],
                            num_heads=num_heads[2],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            # drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                            drop_path=0.,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.msaff2 = MSAFF(dim=dim*4)
        self.dowsample_2 = Downsample(dim*4, dim*8)
        
        # Bottleneck
        self.conv = TRANSFORMER_BLOCK(dim=dim*8,
                            input_resolution=(img_size // (2 ** 3),
                                                img_size // (2 ** 3)),
                            depth=depths[3],
                            num_heads=num_heads[3],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            # drop_path=conv_dpr,
                            drop_path=0.,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        
        # Decoder
        self.upsample_0 = Upsample(dim*8, dim*4)
        self.sctam0 = Pixel_Shuffle(int(dim * 4), up_scale=4)
        self.pifm0 = MFF(int(dim * 2 ** 2),5)
        self.decoderlayer_0 = TRANSFORMER_BLOCK(dim=dim*8,
                            input_resolution=(img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                            depth=depths[4],
                            num_heads=num_heads[4],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            # drop_path=dec_dpr[:depths[4]],
                            drop_path=0.,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.upsample_1 = Upsample(dim*8, dim*2)
        self.sctam1 = Pixel_Shuffle(int(dim * 2), up_scale=2)
        self.pifm1 = MFF(int(dim * 2 ** 1),3)
        self.decoderlayer_1 = TRANSFORMER_BLOCK(dim=dim*4,
                            input_resolution=(img_size // (2 ** 1),
                                                img_size // (2 ** 1)),
                            depth=depths[5],
                            num_heads=num_heads[5],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            # drop_path=dec_dpr[sum(depths[4]):sum(depths[4:6])],
                            drop_path=0.,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.upsample_2 = Upsample(dim*4, dim)
        self.sctam2 = Pixel_Shuffle(int(dim * 1), up_scale=1)
        self.pifm2 = MFF(int(dim),1)
        self.decoderlayer_2 = TRANSFORMER_BLOCK(dim=dim*2,
                            input_resolution=(img_size,
                                                img_size),
                            depth=depths[6],
                            num_heads=num_heads[6],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            # drop_path=dec_dpr[sum(depths[4:6]):sum(depths[4:7])],
                            drop_path=0.,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.patch_unembed = nn.Conv2d(dim*2, out_channel, kernel_size=to_2tuple(3), padding=1, bias=False)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)    
    # 未做消融的架构
    def forward_features(self, x, mask=None):    
        y = self.input_proj(x)
        y = self.pos_drop(y)
        conv0 = self.encoderlayer_0(y,mask=mask)               
        pool0 = self.dowsample_0(conv0)
        
        conv1 = self.encoderlayer_1(pool0,mask=mask)        
        pool1 = self.dowsample_1(conv1)
        
        conv2 = self.encoderlayer_2(pool1,mask=mask)        
        pool2 = self.dowsample_2(conv2)
        
        conv3 = self.conv(pool2, mask=mask)
                
        conv0 = self.msaff0(conv0) 
        conv1 = self.msaff1(conv1) 
        conv2 = self.msaff2(conv2)
        
        up0 = self.upsample_0(conv3)
        _, fake_image_x8 = self.sctam0(up0)        
        
        deconv0 = self.pifm0([up0,conv2, conv0, conv1])
        deconv0 = self.decoderlayer_0(deconv0)
        
        up1 = self.upsample_1(deconv0)
        _, fake_image_x4 = self.sctam1(up1)
             
        deconv1 = self.pifm1([up1,conv1, conv0, conv2])
        deconv1 = self.decoderlayer_1(deconv1)
        
        up2 = self.upsample_2(deconv1)
        _, fake_image_x2 = self.sctam2(up2)
        
        deconv2 = self.pifm2([up2,conv0, conv1, conv2])
        deconv2 = self.decoderlayer_2(deconv2)
        
        y_ =  changeshape(deconv2)
        y_ = self.patch_unembed(y_)
        return y_, fake_image_x8, fake_image_x4, fake_image_x2     
    
    def forward(self, x, only_last=False, mask=None):
        input_ = x
        _, _, h, w = input_.shape

        x, fake_image_x8,fake_image_x4, fake_image_x2 = self.forward_features(x,mask)
        # x = self.forward_features(x,mask)
        # 在通道维度进行分割
        K, B = torch.split(x, [1, 3], dim=1)

        x = K * input_ - B + input_
        # 裁剪张量 x，保留原始输入图像的高度和宽度。
        x = x[:, :, :h, :w]

        if only_last:
            return x
        else:
            return x, fake_image_x8, fake_image_x4, fake_image_x2
            


if __name__ == '__main__':
    x = torch.randn((1, 3, 256, 256)).cuda()
    net = Model().cuda()

    from thop import profile, clever_format
    flops, params = profile(net, (x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)  