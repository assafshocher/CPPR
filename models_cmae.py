# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet18
from patch_resnet import PatchResNet
from pixel_vit import PixelViT
from vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class ContextLessModelWrapperV2(nn.Module):
    def __init__(self, backbone, projector, img_size, patch_size):
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        self.num_patches = img_size // patch_size

    def forward(self, x):
        B, _, _, _ = x.shape
        x = self.backbone(x)
        x = self.projector(x)
        return x.view(B, self.num_patches * self.num_patches, x.shape[-1])


class FlattenTranspose(nn.Module):
    def forward(self, x):
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x.flatten(0, 1)


class UnFlattenP(nn.Module):
    def __init__(self, num_patches):
        super().__init__()
        self.num_patches = num_patches

    def forward(self, x):
        return x.view(x.shape[0] // self.num_patches, self.num_patches, x.shape[-1])


def Projector(f, activation, use_bn=True):
    layers = []

    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        if use_bn:
            layers.append(nn.BatchNorm1d(f[i + 1]))
        if activation == 'relu':
            layers.append(nn.ReLU(True))
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(True))
        elif activation == 'gelu':
            layers.append(nn.GELU())
        elif activation == 'none':
            pass
        else:
            raise ValueError()
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 args=None, contextless_model='base'):
        super().__init__()

        # --------------------------------------------------------------------------

        # MAE encoder specifics
        self.args = args
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        # contextless network
        layers_spec = list(map(int, args.contextless_model_projector_arch.split("-")))
        decoder_output_dim = layers_spec[-1]
        if contextless_model == 'custom_base_norm':
            backbone = torch.nn.Sequential(PatchEmbed(img_size, patch_size, in_chans, embed_dim),
                                           torch.nn.Flatten(0, 1),
                                           nn.BatchNorm1d(embed_dim),
                                           torch.nn.ReLU(inplace=True))
            projector = Projector(layers_spec, 'relu')
            self.contextless_net = ContextLessModelWrapperV2(backbone, projector, img_size, patch_size)
        elif contextless_model == 'resnet':
            self.contextless_net = PatchResNet(embed_dim, patch_size, img_size)
        elif contextless_model == 'vit':
            decoder_input_dim = int(args.contextless_model_projector_arch.split('-')[0])
            backbone = torch.nn.Sequential(PixelViT(img_size=img_size, embed_dim=decoder_input_dim), torch.nn.Flatten(0, 1))
            projector = Projector(layers_spec, 'gelu')
            self.contextless_net = ContextLessModelWrapperV2(backbone, projector, img_size, patch_size)
        elif contextless_model == 'resnet18':
            backbone = torch.nn.Sequential(resnet18(pretrained=False, strides=[1, 2, 2, 1]), FlattenTranspose())
            projector = Projector(layers_spec, 'relu')
            self.contextless_net = ContextLessModelWrapperV2(backbone, projector, img_size, patch_size)
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, decoder_output_dim, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------
        if self.args.linear_eval:
            if self.args.linear_eval_bn:
                bn = torch.nn.BatchNorm1d(embed_dim, affine=False)
            else:
                bn = torch.nn.Identity()
            self.fc_projector = torch.nn.Sequential(bn, torch.nn.Linear(embed_dim, 1000))
            self.cross_entropy = torch.nn.CrossEntropyLoss()

        self.initialize_weights()

        self.loss_var_coeff = self.args.loss_var_coeff
        self.loss_cov_coeff = self.args.loss_cov_coeff

        if args.use_batch_stats:
            self.loss_var_coeff = self.loss_var_coeff / (1 + self.args.batch_patch_ratio)
            self.loss_var_b_coeff = self.args.batch_patch_ratio * self.loss_var_coeff

            self.loss_cov_coeff = self.loss_cov_coeff / (1 + self.args.batch_patch_ratio)
            self.loss_cov_b_coeff = self.args.batch_patch_ratio * self.loss_cov_coeff

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

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
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_eval_loss(self, h, y):
        h = h.detach()
        y_h_pred = self.fc_projector(h)
        loss_y_h = self.cross_entropy(y_h_pred, y)
        return loss_y_h

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask, latent, label):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        B, L, E = pred.shape
        loss_log = LossLog()

        x = pred
        y = self.contextless_net(imgs)

        # invaraince loss
        loss_invar = F.mse_loss(x, y, reduction='none').mean(dim=-1)  # [B, L, E]
        loss_invar = (loss_invar * mask).sum() / mask.sum()
        loss_log.add_loss('loss_invar', self.args.loss_invar_coeff, loss_invar)

        # patchwise variance loss
        std_y = torch.sqrt(y.var(dim=1) + 0.0001)
        loss_var_p = torch.mean(F.relu(1 - std_y))
        loss_log.add_loss('loss_var', self.loss_var_coeff, loss_var_p)

        # batchwise variance loss
        if self.args.use_batch_stats:
            std_y = torch.sqrt(y.var(dim=0) + 0.0001)
            loss_var_b = torch.mean(F.relu(1 - std_y))
            loss_log.add_loss('loss_var_b', self.loss_var_b_coeff, loss_var_b)

        # patchwise cov loss
        y_centered = y - y.mean(dim=1, keepdim=True)
        cov = torch.einsum('ble,blf->bef', y_centered, y_centered).div(L - 1).pow_(2)
        loss_cov_p = (cov.sum() - torch.diagonal(cov, 1, 2).sum()).div(B * E * (E - 1))
        loss_log.add_loss('loss_cov', self.loss_cov_coeff, loss_cov_p)

        # batchwise cov loss
        if self.args.use_batch_stats:
            y_centered = y - y.mean(dim=0, keepdim=True)
            cov = torch.einsum('ble,blf->lef', y_centered, y_centered).div(B - 1).pow_(2)
            loss_cov_b = (cov.sum() - torch.diagonal(cov, 1, 2).sum()).div(L * E * (E - 1))
            loss_log.add_loss('loss_cov_b', self.loss_cov_b_coeff, loss_cov_b)

        if label is not None:
            loss_lin_prob = self.forward_eval_loss(latent[:, 0], label)
            loss_log.add_loss('loss_lin_prob', 1., loss_lin_prob)

        return loss_log.return_loss()

    def forward(self, imgs, mask_ratio=0.75, y=None, mode='train'):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)

        if mode == "eval":
            return self.fc_projector(latent[:, 0])

        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss_dict = self.forward_loss(imgs, pred, mask, latent, y)

        return loss_dict


class LossLog:
    def __init__(self):
        self.loss_name_lst = []
        self.loss_coeff_lst = []
        self.loss_val_lst = []

    def add_loss(self, loss_name, loss_coeff, loss_val):
        self.loss_name_lst.append(loss_name)
        self.loss_coeff_lst.append(loss_coeff)
        self.loss_val_lst.append(loss_val)

    def return_loss(self):
        loss_dict = {'loss': 0.}
        for name_val, coeff_val, loss_val in zip(self.loss_name_lst, self.loss_coeff_lst, self.loss_val_lst):
            loss_dict[name_val + '_unscaled'] = float(loss_val.detach().item())
            curr_loss = loss_val * coeff_val
            loss_dict['loss'] += curr_loss
            loss_dict[name_val] = float(curr_loss.detach().item())
        return loss_dict


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
