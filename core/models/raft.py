import torch
import torch.nn as nn
import torch.nn.functional as F

from update import *
from extractor import BasicEncoder
from corr import CorrBlock
from utils.utils import coords_grid, upflow8
from gma import Attention
from torch import nn, einsum
from einops import rearrange

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass



class CostGlobalEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sk_conv = eval(args.SKII_Block)(324, 128, args.k_conv, args)
        self.proj = nn.Conv2d(256, 128, 1)
        self.ffn = nn.Sequential(
            nn.Conv2d(128, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, 128, 1),
        )
        self.scale = 128 ** -0.5


    def forward(self, corr, k, v):
        '''
        corr: N, 324, H, W
        k: N*H*W, 128, H, W
        v: N*H*W, 128, H, W
        '''
        q = self.sk_conv(corr)
        residual = q

        # self attention: x = self.attn(q, k, v)
        # ========
        q, k = map(lambda t: rearrange(t, 'b (h d) x y -> b h x y d', h=1), (q, k))
        attn = (einsum('b h x y d, b h u v d -> b h x y u v', q, k) * self.scale)
        attn = rearrange(attn, 'b h x y u v -> b h (x y) (u v)').softmax(dim=-1)

        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=1)
        x = einsum('b h i j, b h j d -> b h i d', attn, v)
        _, _, h, w = residual.shape
        x = rearrange(x, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        # ==========
        x = self.proj(torch.cat([x, residual], dim=1))
        x = x + residual

        x = x + self.ffn(x)

        return x



class RAFT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        self.args.alternate_corr = False

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

        if args.cost_encoder_v1:
            self.cost_memory_encoder = eval(args.SKII_Block)(256, 128, args.k_conv, args)
            self.cost_global_encoder = CostGlobalEncoder(args)
            self.to_k = nn.Conv2d(324, 128, 1)
            self.to_v = nn.Conv2d(324, 128, 1)
        elif args.cost_encoder_v2:
            self.cost_encoder = eval(args.SKII_Block)(324, 512, args.k_conv, args)


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])        
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            if self.args.cost_encoder_v2:
                corr = self.cost_encoder(corr)
            elif self.args.cost_encoder_v1:
                k, v = self.to_k(corr), self.to_v(corr)
                cost_global = self.cost_global_encoder(corr, k, v) # corr是q cost_memory是k, v
                corr = torch.cat([cost_global, corr], dim=1)

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions