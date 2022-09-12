import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed


class PatchGrouperViT(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=8, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, 
                 debug_mode=False, num_groups=2):
        super().__init__()

        # --------------------------------------------------------------------------
        self.debug_mode = debug_mode

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.scale = embed_dim ** -0.5

        self.group_tokens = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, embed_dim)) for _ in range(num_groups)])
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        if not debug_mode:
            self.blocks = nn.ModuleList([
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(depth)])
            self.norm = norm_layer(embed_dim)

        self.pred = nn.Linear(embed_dim, 1, bias=True)

        self.q_mlp = nn.Linear(dim, dim, bias=False)
        self.k_mlp = nn.Linear(dim, dim, bias=False)
        # --------------------------------------------------------------------------

        if not self.debug_mode:
            self.initialize_weights()
    

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        for group_token in self.group_tokens:
            torch.nn.init.normal_(group_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.SyncBatchNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


    def forward(self, imgs):
        # embed patches
        x = self.patch_embed(imgs)

        # add pos embed w/o group token
        x = x + self.pos_embed

        # append group token
        group_tokens = [group_token.expand(x.shape[0], -1, -1) for group_token in self.group_tokens]
        x = torch.cat([*group_tokens, x], dim=1)
            
        # apply Transformer blocks
        if not self.debug_mode:
            for blk in self.blocks:
                x = blk(x)

        group_tokens, patch_tokens = x[:, :num_groups, :], x[:, num_groups:, :]

        # cross attention between patch anf group tokens
        q = self.q_mlp(patch_tokens)
        k = self.k_mlp(group_tokens)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        segment_probs = attn.softmax(dim=-1)

        return segment_probs