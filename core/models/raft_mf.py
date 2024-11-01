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
from temporal_aggregation import MFC

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

class RAFT_MF(nn.Module):
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
        self.update_block = MFUpdateBlock(self.args, hidden_dim=hdim)
        self.motion_encoder = BasicMotionEncoder(args)
        if self.args.use_mfc:
            self.temporal_aggregation = MFC(C_in=128, C_out=128)

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


    def forward(self, images, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        T = len(images)
        for i in range(T):
            images[i] = (2 * (images[i] / 255.0) - 1.0).contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmaps = self.fnet(images)
            cnets = self.cnet(images)
        fmaps = [fmaps[i].float() for i in range(T)]

        flow, motion_feature, flow_predictions_list = None, None, []
        for i in range(T-1):
            with autocast(enabled=self.args.mixed_precision):
                net, inp = torch.split(cnets[i], [hdim, cdim], dim=1)
                net, inp = torch.tanh(net), torch.relu(inp)
            corr_fn = CorrBlock(fmaps[i], fmaps[i+1], radius=self.args.corr_radius)
            coords0, coords1 = self.initialize_flow(images[i])

            flow_prev, motion_feature_prev, flow_predictions = flow, motion_feature, []
            if self.args.use_prev_flow: # use flow predicted in previous flow
                flow_init = flow_prev
            if flow_init is not None:
                coords1 = coords1 + flow_init

            for itr in range(iters):
                coords1 = coords1.detach()
                corr = corr_fn(coords1) # index correlation volume
                flow = coords1 - coords0
                with autocast(enabled=self.args.mixed_precision):
                    motion_feature = self.motion_encoder(flow, corr)
                    if self.args.use_mfc:
                        motion_feature = self.temporal_aggregation(motion_feature, motion_feature_prev)
                    net, up_mask, delta_flow = self.update_block(net, inp, motion_feature)

                coords1 = coords1 + delta_flow

                # upsample predictions
                if up_mask is None:
                    flow_up = upflow8(coords1 - coords0)
                else:
                    flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
                flow_predictions.append(flow_up)

            flow_predictions_list.append(flow_predictions)
        
        if test_mode:
            return flow_up
            
        return flow_predictions_list