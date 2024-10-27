import torch
import torch.nn as nn
import timm
import numpy as np
from functools import partial
from einops import rearrange
from torch import nn, einsum
from timm.models.twins import GlobalSubSampleAttn, LocallyGroupedAttn
from timm.layers import Mlp, DropPath, to_2tuple
from timm.models.vision_transformer import Attention as MyAttention
import math



class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, ws=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if ws is None:
            self.attn = Attention(dim, num_heads, False, None, attn_drop, drop)
        elif ws == 1:
            self.attn = GlobalSubSampleAttn(dim, num_heads, attn_drop, drop, sr_ratio)
        else:
            self.attn = LocallyGroupedAttn(dim, num_heads, attn_drop, drop, ws)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.temporal_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x, size, B, T):
        # x: (B*T, h*w, C)
        x = x + self.drop_path(self.attn(self.norm1(x), size))
        # spatial
        temp = self.mlp(self.norm2(x))

        # temporal
        temp = rearrange(temp, '(b t) (h w) c -> (b h w) c t', b=B, t=T, h=size[0], w=size[1])
        temp = self.temporal_conv(temp)
        temp = rearrange(temp, '(b h w) c t -> (b t) (h w) c', b=B, t=T, h=size[0], w=size[1])

        # output
        x = x + self.drop_path(temp)
        return x



class Twins_1DConv(nn.Module):
    def __init__(self, pretrained=True, args=None, **kwargs):
        super().__init__()
        self.svt = timm.create_model('twins_svt_large', pretrained=False)
        self.args = args

        # Replace Block
        self.svt.blocks = nn.ModuleList()
        depths = [2,2,18,2]
        drop_path_rate = 0.1
        embed_dims = [128, 256, 512, 1024]
        num_heads = [4, 8, 16, 32]
        mlp_ratios = [4, 4, 4, 4]
        drop_rate = 0.
        attn_drop_rate = 0.
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        sr_ratios = [8, 4, 2, 1]
        wss = [7, 7, 7, 7]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for k in range(len(depths)):
            _block = nn.ModuleList([Block(
                dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[k],
                ws=1 if wss is None or i % 2 == 1 else wss[k]) for i in range(depths[k])])
            self.svt.blocks.append(_block)
            cur += depths[k]

        if pretrained: self.svt.load_state_dict(torch.load('./pretrained/twins_svt_large-90f6aaa9.pth'), strict=False)

        del self.svt.head
        del self.svt.patch_embeds[2]
        del self.svt.patch_embeds[2]
        del self.svt.blocks[2]
        del self.svt.blocks[2]
        del self.svt.pos_block[2]
        del self.svt.pos_block[2]
        del self.svt.pos_drops[2]
        del self.svt.pos_drops[2]

        # init temporal blocks
        for name, m in self.named_modules():
            if 'temporal_conv' in name:
                nn.init.dirac_(m.weight.data) # initialized to be identity
                nn.init.zeros_(m.bias.data)

        
    def forward(self, x, data=None, layer=2):
        # if input is list, combine batch dimension
        B, T = x.shape[:2]
        x = rearrange(x, 'b t c h w -> (b t) c h w')

        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):

            x, size = embed(x) # x: (B*T, h*w, C)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size, B, T)
                if j == 0:
                    x = pos_blk(x, size)

            # self.svt.depths实质为2
            x = x.reshape(B*T, *size, -1).permute(0, 3, 1, 2).contiguous()

            if i == layer-1:
                break

        x = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)

        return x


