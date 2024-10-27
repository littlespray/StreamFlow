import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

autocast = torch.cuda.amp.autocast
 
def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].expand(batch, -1, -1, -1)

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1) # shape: [9]
            dy = torch.linspace(-r, r, 2 * r + 1) # shape: [9]
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device) # shape: [9, 9, 2]

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())





import torch
import torch.nn as nn
import timm
import numpy as np
from functools import partial
from einops import rearrange
from torch import nn, einsum
from timm.layers import Mlp, DropPath, to_2tuple
import math


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: B, T, C, H, W
        B, T, _, H, W = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w -> b (t h w) c', b=B, t=T)
        x = self.norm(x)
        out_size = ((H*T) // self.patch_size[0], W // self.patch_size[1])

        return x, out_size


class Twins_CSC(nn.Module):
    def __init__(self, pretrained=False, args=None, **kwargs):
        super().__init__()
        self.svt = timm.create_model('twins_svt_large', pretrained=False)
        self.svt.patch_embeds = nn.ModuleList()
        embed_dims = [128, 256, 512, 1024]

        self.svt.patch_embeds.append(PatchEmbed(4, 3, embed_dims[0])) # 4倍下采样
        self.svt.patch_embeds.append(PatchEmbed(2, embed_dims[0], embed_dims[1])) # 2倍下采样
        self.svt.patch_embeds.append(PatchEmbed(2, embed_dims[1], embed_dims[2])) # 2倍下采样
        self.svt.patch_embeds.append(PatchEmbed(2, embed_dims[2], embed_dims[3])) # 2倍下采样
        del self.svt.head
        del self.svt.blocks[2]
        del self.svt.blocks[2]
        del self.svt.pos_block[2]
        del self.svt.pos_block[2]
        del self.svt.patch_embeds[2]
        del self.svt.patch_embeds[2]
    
    def forward(self, x):
        layer = 2
        # if input is list, combine batch dimension
        B, T, C, H, W = x.shape
        ratios = [4, 2]

        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):

            x, size = embed(x) # x: (B, T*h*w, C)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j==0:
                    x = pos_blk(x, size)

            # self.svt.depths实质为2
            H, W = H // ratios[i], W // ratios[i]
            x = rearrange(x, 'b (t h w) c -> b t c h w', t=T, h=H, w=W)

            # x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()

            if i == layer-1:
                break

        return x




class SKBlock(nn.Module):
    def __init__(self, C_in, C_out, k_conv):
        super().__init__()
        self.conv_list = nn.ModuleList([
            nn.Conv2d(C_in, C_in, kernel, stride=1, padding=kernel//2, groups=C_in) for kernel in k_conv])

        self.ffn1 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5*C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5*C_in), C_in, 1, padding=0),
        )
        self.pw = nn.Conv2d(C_in, C_in, 1, padding=0)
        self.ffn2 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5*C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5*C_in), C_out, 1, padding=0),
        )

    def forward(self, x):
        x = F.gelu(x + self.ffn1(x))
        for conv in self.conv_list:
            x = F.gelu(x + conv(x))
        x = F.gelu(x + self.pw(x))
        x = self.ffn2(x)
        return x


class SKMotionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        out_dim = 128
        cor_planes = 4* (2*4 + 1)**2

        self.convc1 = SKBlock(cor_planes, 256, [1,15])
        self.convc2 = SKBlock(256, 192, [1,15])

        self.convf1 = nn.Conv2d(2, 128, 1, 1, 0)
        self.convf2 = SKBlock(128, 64, [1,15])

        self.conv = SKBlock(64+192, out_dim-2, [1,15])


    def forward(self, flow, corr, attention=None):
        cor = F.gelu(self.convc1(corr))
        cor = self.convc2(cor)

        flo = self.convf1(flow)
        flo = self.convf2(flo)

        cor_flo = torch.cat([cor, flo], dim=1)
        out = self.conv(cor_flo)

        return torch.cat([out, flow], dim=1)

# Agg
class Aggregate(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 128,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))



    def forward(self, querys, keys, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        v = self.to_v(fmap)        
        v = rearrange(v, 'b (h d) x y -> b (x y) h d', h=heads)
        # out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        
        out = flash_attn_func(querys, keys, v, dropout_p=0.0, softmax_scale=self.scale, causal=False)
        out = rearrange(out, 'b (x y) h c -> b (h c) x y', h=heads,x=h,y=w)


        out = fmap + self.gamma * out

        return out



class Attention(nn.Module):
    def __init__(
        self,
        dim,
        max_pos_size = 100,
        heads = 4,
        dim_head = 128,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qk = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)
        # self.pos_emb = RelPosEmb(max_pos_size, dim_head)

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        q, k = self.to_qk(fmap).chunk(2, dim=1) # b (head dim) x y
        q, k = map(lambda t: rearrange(t, 'b (h d) x y -> b (x y) h d', h=heads), (q, k))

        # attn = flash_attn_func(q, k, fmap, dropout_p=0.0, softmax_scale=self.scale, causal=False)
        
        return q, k



# TemporalLayer2
def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module

from timm.models.vision_transformer import Attention as timm_attn
from timm.layers import DropPath, Mlp
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=1, mlp_ratio=2, drop_rate=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path1 = DropPath(drop_rate)
        self.drop_path2 =DropPath(drop_rate)
        
        self.attn = timm_attn(
            dim,
            num_heads=num_heads,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.GELU,
        )
        self.mlp = Mlp(
                    in_features=dim,
                    hidden_features=int(dim * mlp_ratio),
                    act_layer=nn.GELU,
                    drop=0.,)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x
class TemporalLayer2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.transformer_block = zero_module(TransformerBlock(dim))
        
    def forward(self, x, HW):
        # input: (B) (T H W) C
        # output: (B T) C H W
        H, W = HW[0], HW[1]
        x = self.transformer_block(x)
        x = rearrange(x, '(b h w) t c -> (b t) c h w', h=H, w=W)
        return x

class StreamFlowUpdateBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SKMotionEncoder()
        ratio = 8 
        embed_dim = 128

        self.aggregator = Aggregate(dim=embed_dim, dim_head=embed_dim, heads=1)
        self.gru = SKBlock(embed_dim*5, embed_dim, k_conv=[1, 7])


        self.mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim*2, ratio*ratio*9, 1, padding=0))
        
        self.transformer_block = TemporalLayer2(dim=embed_dim)
        self.flow_head = SKBlock(embed_dim*(4-1), 2 * (4-1), [1,15])

    def forward(self, nets, inps, corrs, flows, querys, keys, T=None):
        BT, _, H, W = nets.shape
        B = BT // T

        motion_features = self.encoder(flows, corrs)
        motion_features_globals = self.aggregator(querys, keys, motion_features)
        motion_features_temporal = self.transformer_block(rearrange(motion_features, '(B T) C H W -> (B H W) T C', T=T), HW=(H, W))
        inp_cats = torch.cat([inps, motion_features, motion_features_globals, motion_features_temporal], dim=1)
        # Temporal Attention
        # Attentional update
        nets = self.gru(torch.cat([nets, inp_cats], dim=1))
        delta_flows = self.flow_head(rearrange(nets, '(B T) C H W -> B (T C) H W', T=T)) # (b t) c h w => b (t c) h w

        # scale mask to balence gradients
        masks = .25 * self.mask(nets)

        masks = rearrange(masks, '(B T) C H W -> B T C H W', B=B, T=T)
        delta_flows = rearrange(delta_flows, 'B (T C) H W -> B T C H W ', T=T)

        return nets, masks, delta_flows


class StreamFlowT4(nn.Module):
    def __init__(self):
        super().__init__()

        self.context_dim = cdim = 128
        self.hidden_dim = 128
        # feature network, context network, and update block
        self.fnet = Twins_CSC()
        self.cnet = Twins_CSC()
        self.update_block = StreamFlowUpdateBlock()
        self.att = Attention(dim=cdim, heads=1, max_pos_size=160, dim_head=cdim)
        self.ratio = 8
        #ckpt = '/apdcephfs_cq10/shangkunsun/StreamFlow2/streamflow_t4-things.pth'
        #dic = {k[7:]:v for k, v in torch.load(ckpt)['model'].items()}
        #self.load_state_dict(dic, strict=True)
        self.freeze_all()


    def freeze_all(self):
        for name, param in self.named_parameters():
            param.requires_grad = False


    def initialize_flow(self, img, ratio=8):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // ratio, W // ratio).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0

    def upsample_flow(self, flow, mask, ratio=8):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, ratio, ratio, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(ratio * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, ratio * H, ratio * W)




    def forward(self, images, iters=12, flow_init=None, upsample=True, test_mode=True):
        """ Estimate optical flow between pair of frames """
        B, T, C, H, W = images.shape

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=True):
            fmaps = self.fnet(images).float()
            cnets = self.cnet(images[:,:-1])

        corr_fns = [CorrBlock(fmaps[:,i], fmaps[:,i+1], radius=4) for i in range(T-1)]
        coord_0s = [self.initialize_flow(images[:,i], ratio=self.ratio) for i in range(T-1)]
        coord_1s = [self.initialize_flow(images[:,i], ratio=self.ratio) for i in range(T-1)]

        if flow_init is not None:
            coord_1s = [coord_1s[i] + flow_init[i] for i in range(len(flow_init))]

        nets, inps, attentions = [], [], []
        with autocast(enabled=True):
            nets, inps = torch.split(cnets, [hdim, cdim], dim=2)
            nets = torch.tanh(rearrange(nets, 'B T C H W -> (B T) C H W'))
            inps = torch.relu(inps)
            inps = rearrange(inps, 'B T C H W -> (B T) C H W')
            if self.att is not None:
                querys, keys = self.att(inps)
            else:
                attentions = None
            

        flow_predictions_list = [[] for i in range(T-1)]
        for itr in range(iters):
            coord_1s = [coord.detach() for coord in coord_1s]
            corrs = rearrange(torch.stack([corr_fns[i](coord_1s[i]) for i in range(T-1)], dim=1), 'B T C H W -> (B T) C H W')
            flows = rearrange(torch.stack([coord_1s[i] - coord_0s[i] for i in range(T-1)], dim=1), 'B T C H W -> (B T) C H W')

            with autocast(enabled=True):
                nets, up_masks, delta_flows = self.update_block(nets, inps, corrs, flows, querys, keys, T=T-1)
            
            coord_1s = [coord_1s[i] + delta_flows[:, i] for i in range(T-1)]
            for i in range(T-1):
                flow_predictions_list[i].append(self.upsample_flow(coord_1s[i] - coord_0s[i], up_masks[:, i], ratio=self.ratio))

        if test_mode:
            if flow_init is None:
                return [flow_predictions[-1] for flow_predictions in flow_predictions_list]
            else:
                flows_lowres = [coord_1s[i] - coord_0s[i] for i in range(T-1)]
                return [flow_predictions[-1] for flow_predictions in flow_predictions_list], flows_lowres
            
        return flow_predictions_list
            


if __name__ == '__main__':
    x = torch.randn(1, 4, 3, 1440, 2560).cuda()
    with torch.no_grad():
        model = StreamFlowT4().cuda()
        while True:
            y = model(x)

