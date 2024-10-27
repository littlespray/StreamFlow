import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention as timm_attn

from update import *
from encoders import *
from corr import CorrBlock
from utils.utils import coords_grid
from gma import Attention, TemporalAttention

from timm.models.layers import DropPath, Mlp

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



class SKFlow_MF8(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.decoder_dim is None: args.decoder_dim = 256
        self.context_dim = cdim = args.decoder_dim // 2
        self.hidden_dim = args.decoder_dim // 2
        args.corr_levels = 4
        args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        # feature network, context network, and update block
        self.fnet = eval(args.Encoder)(args, norm_fn='instance')
        self.cnet = eval(args.Encoder)(args, norm_fn='batch')
        self.update_block = eval(args.UpdateBlock)(self.args)
        if args.use_gma:
            self.att = Attention(args=self.args, dim=cdim, heads=self.args.num_heads, max_pos_size=160, dim_head=cdim)
        else:
            self.att = None
        # self.temp_att = Attention(args=self.args, dim=cdim, heads=self.args.num_heads, max_pos_size=160, dim_head=cdim)
        self.ratio = 16 if args.Encoder == 'UMT' else 8


    def freeze_untemporal(self):
        for name, param in self.named_parameters():
            if "temporal" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_parameters(self):
        for name, param in self.named_parameters():
            param.requires_grad = True


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


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

    def forward(self, images, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """
        T = len(images)
        images = torch.stack(images, dim=1) # B, T, C, H, W
        B, T, C, H, W = images.shape
        images = 2 * (images / 255.0) - 1.0

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmaps = self.fnet(images).float()
            cnets = self.cnet(images[:,:-1])

        corr_fns = [CorrBlock(fmaps[:,i], fmaps[:,i+1], radius=self.args.corr_radius) for i in range(T-1)]
        coord_0s = [self.initialize_flow(images[:,i], ratio=self.ratio) for i in range(T-1)]
        coord_1s = [self.initialize_flow(images[:,i], ratio=self.ratio) for i in range(T-1)]

        if flow_init is not None:
            coord_1s = [coord_1s[i] + flow_init[i] for i in range(len(flow_init))]

        nets, inps, attentions = [], [], []
        with autocast(enabled=self.args.mixed_precision):
            nets, inps = torch.split(cnets, [hdim, cdim], dim=2)
            nets = torch.tanh(rearrange(nets, 'B T C H W -> (B T) C H W'))
            inps = torch.relu(inps)
            inps = rearrange(inps, 'B T C H W -> (B T) C H W')
            if self.att is not None:
                attentions = self.att(inps)
            else:
                attentions = None
            

        flow_predictions_list = [[] for i in range(T-1)]
        for itr in range(iters):
            coord_1s = [coord.detach() for coord in coord_1s]
            corrs = rearrange(torch.stack([corr_fns[i](coord_1s[i]) for i in range(T-1)], dim=1), 'B T C H W -> (B T) C H W')
            flows = rearrange(torch.stack([coord_1s[i] - coord_0s[i] for i in range(T-1)], dim=1), 'B T C H W -> (B T) C H W')

            with autocast(enabled=self.args.mixed_precision):
                nets, up_masks, delta_flows = self.update_block(nets, inps, corrs, flows, attentions, T=T-1)
            
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
