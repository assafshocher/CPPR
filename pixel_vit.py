from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from patch_resnet import PatchResNet
from vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed

class PixelViT(nn.Module):
    def __init__(self, img_size=16, big_patch_size=16, patch_size=2, in_chans=3,
                 embed_dim=256, depth=4, num_heads=4,
                 mlp_ratio=2., norm_layer=nn.LayerNorm, out_chans=768):
        super().__init__()

        self.out_chans = out_chans
        self.num_big_patches = (img_size // big_patch_size) ** 2
        self.p_sz = big_patch_size

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
 
        self.final_linear = nn.Linear(embed_dim, out_chans, bias=True)  # decoder to patch
       

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (B, C, H, H)
        x: (B*(H//P)**2, C, P, P)
        """
        P = self.p_sz
        B, C, H, W = imgs.shape
        assert H == W and H % P == 0

        h = H // P
        x = imgs.reshape(B, C, h, P, h, P)
        x = torch.einsum('bchpwq->bhwcpq', x)
        x = x.reshape(B*h*h, C, P, P)
        return x

    def forward(self, x):
        b_sz = x.shape[0]
        # big img to big patches [B, C, H, W] -> [B*(H//P)**2, C, P, P]
        x = self.patchify(x)

        # embed tiny patches [B*(H//P)**2, C, P, P] -> [B*(H//P)**2, (P//p)**2, C*(p**2)]
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        # add pos embed w/o cls token
        x = x + self.pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # final projection from cls token
        x = self.final_linear(x[:, 0, :])

        return x.view(b_sz, self.num_big_patches, self.out_chans)