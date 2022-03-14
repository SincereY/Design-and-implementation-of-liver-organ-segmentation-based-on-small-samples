import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
import math


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, D ,H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.LeakyReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, D, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, D, H, W) # conv3d 提供位置信息
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=2):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # self.qc = nn.Conv3d(dim, dim, kernel_size=1, dilation=1, groups=dim)
        # self.d1 = nn.Conv3d(dim, dim, kernel_size=3, dilation=1, padding=1, groups=dim)
        # self.d3 = nn.Conv3d(dim, dim, kernel_size=3, dilation=3, padding=3, groups=dim)
        # self.d5 = nn.Conv3d(dim, dim, kernel_size=3, dilation=5, padding=5, groups=dim)
        # self.ln = nn.LayerNorm(dim)

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)


    def forward(self, x, D, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # x1 = x.permute(0, 2, 1).reshape(B, C, D, H, W)
        # # print(x1.shape)
        # q_c = self.qc(x1)
        # # print(q_c.shape)
        # d_1 = self.d1(q_c)
        # d_3 = self.d3(q_c)
        # d_5 = self.d5(q_c)
        # d_ = (d_1 + d_3 + d_5).reshape(B, C, -1).permute(0, 2, 1)
        # q = self.ln(d_).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, D, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.LeakyReLU, norm_layer=nn.LayerNorm, sr_ratio=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, D ,H, W):
        x = x + self.attn(self.norm1(x), D, H, W)
        x = x + self.mlp(self.norm2(x), D, H, W)

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=(8,8,8), patch_size=(3,3,3), in_chans=320, embed_dim=320, dropout = 0.):
        super().__init__()

        self.patch_size = patch_size
        self.D, self.H , self.W = img_size[0] // patch_size, img_size[1] // patch_size, img_size[2] // patch_size
        self.num_patches = self.H * self.W * self.D
        self.num_patches = img_size[0]*img_size[1]*img_size[2]
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(3,3,3), padding=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))  # (1, 512, 768) 这是个偏置
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        # x:(B, C, D ,H, W)

        x = self.proj(x)

        _, _, D ,H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # x:(B, N ,C)


        x = x + self.position_embeddings
        x = self.dropout(x)
        x = self.norm(x)

        return x, D ,H, W


class ShiftedPatchTokenization(nn.Module):
    def __init__(self,img_size, in_dim, dim, dropout = 0., merging_size=2, is_pe=False):
        super().__init__()

        self.patch_shifting = PatchShifting(merging_size)
        self.num_patches = img_size[0] * img_size[1] * img_size[2]
        patch_dim = (in_dim * 9)
        # print(patch_dim)
        self.is_pe = is_pe

        self.merging = nn.Sequential(
            Rearrange('b c h w d -> b (h w d) c'),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches, dim))

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        _, _, D, H, W = x.shape

        out = x if self.is_pe else rearrange(x, 'b (h w d) c -> b c h w d', h=int(math.sqrt(x.size(1))),
                                             w=int(math.sqrt(x.size(2))))
        # print(out.shape)
        out = self.patch_shifting(out)
        # print(out.shape)
        out = self.merging(out)
        # print(out.shape)
        # print(self.position_embeddings.shape)
        out = out + self.position_embeddings
        out = self.dropout(out)
        out = self.norm(out)

        return out, D ,H, W


class PatchShifting(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.shift = int(patch_size * (1 / 2))

    def forward(self, x):
        # print(x.shape)
        x_pad = torch.nn.functional.pad(x, (self.shift, self.shift, self.shift, self.shift, self.shift, self.shift))
        # print(x_pad.shape)
        # if self.is_mean:
        #     x_pad = x_pad.mean(dim=1, keepdim = True)

        """ 4 cardinal directions """
        #############################
        # x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
        # x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
        # x_t2 = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
        # x_b2 = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
        # x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2], dim=1)
        #############################

        """ 4 diagonal directions """
        # #############################
        # x_lu = x_pad[:, :, :-self.shift * 2, :-self.shift * 2]
        # x_ru = x_pad[:, :, :-self.shift * 2, self.shift * 2:]
        # x_lb = x_pad[:, :, self.shift * 2:, :-self.shift * 2]
        # x_rb = x_pad[:, :, self.shift * 2:, self.shift * 2:]
        # x_cat = torch.cat([x, x_lu, x_ru, x_lb, x_rb], dim=1)
        # #############################

        """ 8 cardinal directions """
        #############################
        # x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
        # x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
        # x_t2 = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
        # x_b2 = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
        # x_lu = x_pad[:, :, :-self.shift*2, :-self.shift*2]
        # x_ru = x_pad[:, :, :-self.shift*2, self.shift*2:]
        # x_lb = x_pad[:, :, self.shift*2:, :-self.shift*2]
        # x_rb = x_pad[:, :, self.shift*2:, self.shift*2:]
        # x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2, x_lu, x_ru, x_lb, x_rb], dim=1)
        #############################

        x1 = x_pad[:, :, :-self.shift * 2, :-self.shift * 2, :-self.shift * 2]
        x2 = x_pad[:, :, :-self.shift * 2, :-self.shift * 2, self.shift * 2:]
        x3 = x_pad[:, :, :-self.shift * 2, self.shift * 2:, :-self.shift * 2]
        x4 = x_pad[:, :, :-self.shift * 2, self.shift * 2:, self.shift * 2:]
        x5 = x_pad[:, :, self.shift * 2:, :-self.shift * 2, :-self.shift * 2]
        x6 = x_pad[:, :, self.shift * 2:, :-self.shift * 2, self.shift * 2:]
        x7 = x_pad[:, :, self.shift * 2:, self.shift * 2:, :-self.shift * 2]
        x8 = x_pad[:, :, self.shift * 2:, self.shift * 2:, self.shift * 2:]
        x_cat = torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8], dim=1)

        # out = self.out(x_cat)
        out = x_cat
        # print(out.shape)
        return out


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size, in_chans=1, embed_dims=320, depths = 6,
                 num_heads=4, mlp_ratios=2, qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., norm_layer=nn.LayerNorm,
                 sr_ratios=2):
        super().__init__()

        # patch_embed
        # self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=2, in_chans=in_chans, embed_dim=embed_dims)
        self.patch_embed2 = ShiftedPatchTokenization(img_size, in_chans, embed_dims, drop_rate, 2, is_pe=True)

        # transformer encoder
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer,
            sr_ratio=sr_ratios) for j in range(depths)])
        self.norm1 = norm_layer(embed_dims)

    def forward_features(self, x):
        B = x.shape[0]

        # stage 1
        # x, D, H, W = self.patch_embed1(x)
        print(x.shape)
        x, D, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, D, H, W)
        x = self.norm1(x)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


if __name__ == "__main__":
    input_arr = torch.rand((2,320, 6, 7, 7))
    print(input_arr.shape)
    transform = MixVisionTransformer(img_size=(6,7,7), in_chans=320, embed_dims=320)
    output = transform(input_arr)
    print(output.shape)


