import timm
import torch
import torch.nn as nn
from timm.layers.format import Format

class PatchEmbed(nn.Module):
    def __init__(
            self,
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            flatten=True,
            output_fmt=None,
            bias=True,
            strict_img_size=True,
    ):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.img_size = img_size
        if img_size is not None:
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x


class SwinV2(nn.Module):
    def __init__(self, args=None, pretrained=True, **kwargs):
        super().__init__()
        self.model = timm.create_model('swinv2_base_window12to16_192to256', pretrained=False)
        self.model.patch_embed = PatchEmbed(img_size=(448, 768), patch_size=4,
            in_chans=3, embed_dim=128, output_fmt='NHWC')

        ckpt = torch.load('swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth')['model']
        model_keys = self.model.state_dict().keys()
        ckpt_keys = ckpt.keys()
        extra_keys = model_keys - ckpt_keys
        missing_keys = ckpt_keys - model_keys
        print('model多出来的', extra_keys)
        print('model没有的', missing_keys)
        self.model.load_state_dict(ckpt)
        # print(self.model.state_dict().keys())
        # self.model.load_state_dict({('model.'+k): v for k, v in ckpt.items()}, strict=True)

        self.args = args

        del self.model.head
        del self.model.norm
        del self.model.layers[2]
        del self.model.layers[2]

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')

        x = self.model.patch_embed(x)
        x = self.model.layers(x)

        x = rearrange(x, "(b t) c h w -> b t c h w", t=T)

        return x

if __name__ == '__main__':
    # Build or load a model, e.g. timm's pretrained resnet18
    model = SwinV2().cuda()
    a = model(torch.rand(1, 16, 3, 448, 768).cuda())

    print(a.shape)



