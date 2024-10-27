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




class Twins_3DConv(nn.Module):
    def __init__(self, pretrained=True, args=None, **kwargs):
        super().__init__()
        self.svt = timm.create_model('twins_svt_large', pretrained=False)
        self.args = args

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

        # initial temporal conv
        self.temporal_conv = nn.Conv3d(256, 256, kernel_size=(3,3,3), padding=(1,1,1))
        self.temporal_drop_path = DropPath(0.01) # if drop_path > 0. else nn.Identity()

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
                x = blk(x, size)
                if j == 0:
                    x = pos_blk(x, size)

            # self.svt.depths实质为2
            x = x.reshape(B*T, *size, -1).permute(0, 3, 1, 2).contiguous()

            if i == layer-1:
                break

        # temporal
        x = rearrange(x, '(b t) c h w -> b c t h w', b=B, t=T, h=size[0], w=size[1])
        x = x + self.temporal_drop_path(self.temporal_conv(x))
        x = rearrange(x, 'b c t h w -> b t c h w', b=B, t=T, h=size[0], w=size[1])

        return x