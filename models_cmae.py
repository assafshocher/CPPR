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
from einops import rearrange

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, 
                 use_cls_token=True, slim_predictor=True, cls_predict_loss=False,
                 debug_mode=False,  num_groups=2, temperature=0.1,
                 coeff_ginvar=1., coeff_bvar=1., coeff_pvar=1., coeff_fcov=1., coeff_pcross=1., 
                 coeff_var_thr=1., detach=False):
        super().__init__()

        # --------------------------------------------------------------------------
        self.use_cls_token = use_cls_token
        self.slim_predictor = slim_predictor
        self.cls_predict_loss = cls_predict_loss
        self.debug_mode = debug_mode
        self.temperature = temperature
        self.w_batchwise_loss = w_batchwise_loss
        self.w_patchwise_loss = w_patchwise_loss
        self.w_pred_loss = w_pred_loss
        self.criterion = torch.nn.CrossEntropyLoss()
        self.detach = detach

        self.coeff_ginvar = coeff_ginvar  # group invariance (equivalent to MSE for G=2)
        self.coeff_bvar = coeff_bvar  # batchwise variance
        self.coeff_pvar = coeff_pvar  # patchwise variance
        self.coeff_fcov = coeff_fcov  # feature covariance
        self.coeff_pcross = coeff_pcross  # feature X pos_embd cross-covariance
        self.coeff_var_thr = coeff_var_thr
        self.use_contextless = False

        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + use_cls_token, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        if not debug_mode:
            self.blocks = nn.ModuleList([
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(depth)])
            self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        if cls_predict_loss:
            self.cls_predict_tokens_mlp = nn.Sequential(nn.Linear(num_patches, embed_dim, bias=True),
                                                        nn.GELU(), 
                                                        nn.Linear(embed_dim, embed_dim, bias=True))

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + use_cls_token, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        if not debug_mode:
            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(decoder_depth)])

            self.decoder_norm = norm_layer(decoder_embed_dim)
            self.decoder_pred = nn.Linear(decoder_embed_dim, embed_dim, bias=True) # decoder to embedding
        # --------------------------------------------------------------------------
        self.fc_for_cross_corr_reps = nn.Sequential(nn.Linear(embed_dim, embed_dim, bias=True),
                                                    nn.GELU(), 
                                                    nn.Linear(embed_dim, embed_dim, bias=True))

        self.fc_for_cross_corr_embs = nn.Sequential(nn.Linear(embed_dim, embed_dim, bias=True),
                                                    nn.GELU(),
                                                    nn.Linear(embed_dim, embed_dim, bias=True),
                                                    nn.GELU())


        self.fc_projector = torch.nn.Linear(embed_dim, 1000)
        self.fc_projector = torch.nn.Sequential(torch.nn.BatchNorm1d(embed_dim, affine=False), self.fc_projector)
        self.cross_entropy = torch.nn.CrossEntropyLoss()


        if not self.debug_mode:
            self.initialize_weights()
    
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
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
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.SyncBatchNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


    def random_divide(self, x, num_groups, group_sz, group_duplicates):
        """
        x: [B*G, L, E] if group_duplicates else [B, L, E]
        x_out: [B, G, S, E]
        mask: [B, G, L], indicator along G-dim which group this patch belongs to, all 0 means no-one.
        """
        A, L, E = x.shape  # A might be B*G or just B
        G = num_groups
        S = group_sz

        # handle shapes
        if group_duplicates:
            B = A // G
            x = x.view(B, G, L, E)
        else:
            B = A
            x = x.unsqueeze(1).expand(B, G, L, E)
       
        assert G * S <= L

        # create a random binary mask per group
        perm = torch.randperm(L, device=x.device)[None, :]
        lower_bds = S * torch.arange(G, device=x.device)[:, None]
        mask = (perm >= lower_bds) * (perm < lower_bds + S)[None, ...].expand(B, G, L)

        # mask embds without pos embeding
        embds_no_pos_masked = x[mask]  # [B, G, S, E]
        
        # add pos embd
        x = x + self.pos_embed[:, 1:, :]

        # mask embds with pos embeding
        embds_with_pos_masked = x[mask].view(B, G, S, E)
        
        return embds_with_pos_masked, embds_no_pos_masked, mask

    def forward_encoder(self, x):
        """
        patches_embeddings_divided: [B, G, S, E]
        """
        B, G, S, E = x.shape
        x = x.view(B*G, S, E)

        # append cls token
        if self.use_cls_token:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)  # x: [B*G, 1+S, E]

        # apply Transformer blocks
        if not self.debug_mode:
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)

        x = x.view(B, G, S+self.use_cls_token, E)

        return x

        
    def arrange_reps(self, reps_divided, embds_no_pos_divided, mask):
        """
        reps_divided: [B, G, S+1, E] if self.use_cls_token else [B, G, S, E]
        mask: [B, G, L], indicator along G-dim which group this patch belongs to, all 0 means no-one.
        """
        B, G, S, E = reps_divided.shape  # if cls token used then this S is wrong, will be fixed below 
        L = mask.shape[-1]
        
        # take care of the cls token
        if self.use_cls_token:
            S = S - 1  # fix S because we intialy added the cls token to the count
            orig_cls_token = reps_divided[:, :, 0, :]  # [B, G, E]
            reps_divided = reps_divided[:, :, 1:, :]
            cls_tokens = orig_cls_token.unsqueeze(2)

        # expand the reps and embds with mask tokens for the missing patches in each group
        expanded_mask_token = self.mask_token.repeat(B, G, L, 1)
        reps_divided_expanded = expanded_mask_token
        reps_divided_expanded[mask] = reps_divided.reshape(-1, E)  # [B, G, L, E]
        embds_no_pos_divided_expanded = expanded_mask_token
        embds_no_pos_divided_expanded[mask] = embds_no_pos_divided.reshape(-1, E).to(embds_no_pos_divided_expanded.dtype)  # [B, G, L, E]
        
        # unite the divided reps by summing along G (only one element is not multiplied by 0)
        reps_united = (reps_divided_expanded * mask.unsqueeze(-1)).sum(1)  # [B, L, E]
        embds_contextless_united = (embds_no_pos_divided_expanded * mask.unsqueeze(-1)).sum(1)  # [B, L, E]
        
        # in this config we throw away mask tokens for patches that are not in any group
        if self.slim_predictor:
            global_mask = mask[0].bool().any(0)  # [L]
            reps_divided_expanded = reps_divided_expanded[:, :, global_mask, :]  # [B, G, S, E]
            reps_united = reps_united[:, global_mask, :]  # [B, G*S, E]
            embds_contextless_united = embds_contextless_united[:, global_mask, :]  # [B, G*S, E]  

        # add the mask tokens in the beginning (adds 0 or G more tokens)
        if self.use_cls_token:
            reps_divided_expanded = torch.cat([cls_tokens, reps_divided_expanded],
                                                         2)  # [B, G, G+P, E]
            if not self.cls_predict_loss:
                orig_cls_token = orig_cls_token.unsqueeze(2).mean(1)
            reps_united = torch.cat([orig_cls_token, reps_united], 1)  # [B, 1+P, E]


        return reps_divided_expanded, reps_united, embds_contextless_united


    def forward_predictor(self, reps_divided, mask):
        """
        reps_divided: [B, G, C+P, E]
        """
        B, G, C_plus_P, E = reps_divided.shape
        C = (G - 1) * self.cls_predict_loss + self.use_cls_token
        P = C_plus_P - C

        # move groups to batch dim to apply blocks
        x = reps_divided.view(B*G, C+P, E)

        if not self.debug_mode:
            # embed tokens
            x = self.decoder_embed(x)

            # add pos embed (need to slim it)
            L = self.decoder_pos_embed.shape[1] - 1
            indices = mask.sum(1, keepdim=True).expand(B, G, L).reshape(B*G, L)
            indices = torch.cat([torch.ones(B*G, 1, device=indices.device), indices], 1)  # pos embed for cls token
            pos_embed = self.decoder_pos_embed.expand(B*G, -1, -1)
            x = x + pos_embed[indices.bool()].view(B*G, 1+P, -1)  # note E is encoder embed sz so we used -1 here.

            # apply Transformer blocks
            for blk in self.decoder_blocks:
                x = blk(x)
            x = self.decoder_norm(x)

            # predictor projection
            x = self.decoder_pred(x)

        pred = x.view(B, G, C+P, E)

        return pred


    def forward_loss(self, pred, rep, mask, rep_contextless):
        """
        pred: [B, G, C+P, E]
        rep: [B, C+P, E]
        rep_contextless: [B, P, E]
        mask: [B, G, L]
        """
        B, G, C_plus_P, E = pred.shape
        C = (G - 1) * self.cls_predict_loss + self.use_cls_token
        P = C_plus_P - C
        L = mask.shape[-1]

        preds = pred

        if self.use_contextless:
            # project contextless representations by MLP on pixels
            rep_contextless = self.fc_for_cross_corr_embs(rep_contextless)  # [B, P, E]
            # cat contextless as another group
            preds = torch.cat(preds[:, :, C:, :], rep_contextless.unsqueeze(1))  # [B, G+1, P, E]

        # groupwise invariance loss
        loss_groupwise_invar = preds.var(1).mean()  # if G==2 equivalent to MSE

        # batchwise variance loss
        std_batchwise = (preds.var(0) + 0.0001).sqrt()
        loss_batchwise_var = F.relu(self.coeff_var_thr - std_batchwise).mean()

        # patchwise variance loss
        std_patchwise = (preds.var(2) + 0.0001).sqrt()
        loss_patchwise_var = F.relu(self.coeff_var_thr - std_patchwise).mean()       

        # featurewise covariance loss
        N = B * (G + self.use_contextless) * (P + C)
        preds_flat = preds.view(N, E)
        preds_flat = preds_flat - preds_flat.mean(0)
        preds_flat = preds_flat / ((N - 1.) ** 0.5)  # numerical stability instead of dividing after
        cov = (preds_flat.T @ preds_flat)  # [E, E]
        off_diag = cov.flatten()[:-1].view(E - 1, E + 1)[:, 1:].flatten()
        loss_featurewise_cov = off_diag.pow(2).sum() / (E ** 2 - E)

        # mask pos embds  (assume here self.use_contextless is False)
        N = B * G * P
        global_mask = mask[0].bool().any(0)  # [L]
        global_mask = torch.cat([torch.zeros(1, dtype=bool, device=pred.device), global_mask])  # [L+1] throw away cls token pos embd
        enc_pos_embed = self.pos_embed[:, None, global_mask, :].repeat(B, G, 1, 1).view(N, E)
        pred_pos_embed = self.decoder_pos_embed[:, None, global_mask, :].repeat(B, G, 1, 1).view(N, -1)  # -1 because it's predictor embd dim

        # centering 
        enc_pos_embed = enc_pos_embed - enc_pos_embed.mean(1, keepdim=True)
        pred_pos_embed = pred_pos_embed - pred_pos_embed.mean(1, keepdim=True)

        # create matrices
        preds_flat_no_cls = preds[:, :, C:, :].reshape(N, E)
        preds_flat_no_cls = preds_flat_no_cls - preds_flat.mean(0)

        # scale to prevent inf, insterad of dividing after composing the matrices
        s = (N - 1.) ** 0.5
        preds_flat_no_cls, enc_pos_embed, pred_pos_embed = preds_flat_no_cls / s , enc_pos_embed / s, pred_pos_embed / s

        cross_cov_pos_enc = (preds_flat_no_cls.T @ enc_pos_embed) # [E, E]
        cross_cov_pos_pred = (preds_flat_no_cls.T @ pred_pos_embed) # [E, Ep]  Ep is embd size in predictor
        loss_pos_cross_cov = 0.5 * (cross_cov_pos_enc.pow(2).sum() + cross_cov_pos_pred.pow(2).sum()) / (E ** 2)

        # combine all the losses
        loss = (self.coeff_ginvar * loss_groupwise_invar + 
                self.coeff_bvar * loss_batchwise_var +
                self.coeff_pvar * loss_patchwise_var +
                self.coeff_fcov * loss_featurewise_cov +
                self.coeff_pcross * loss_pos_cross_cov)
        
        return loss, loss_groupwise_invar, loss_batchwise_var, loss_patchwise_var, loss_featurewise_cov, loss_pos_cross_cov


    def forward_eval_loss(self, h, y):
        h = h.detach()
        y_h_pred = self.fc_projector(h)
        loss_y_h = self.cross_entropy(y_h_pred, y)
        return loss_y_h


    def forward(self, imgs, num_groups, group_sz, y=None, group_duplicates=True, mode='train'):
        """
        B: Batch-size
        in_chans, H, W: channels (typically 3), and spatial dims (typically 224X224)
        L: number of patches in an image (typically 196)
        G: number of groups (typically 2-4)
        S: number of patches in a group. G*S<=L (you don't have to use all patches)
        E: embedding size
        P: number of patch tokens in predictor: G*S if slim_predictor else L
        C: number of class tokens in predictor: G if use_cls_token else 0
        """
        # imgs: [B, in_chans, H, W]

        # first processing of the images to embedded patches
        patch_embds = self.patch_embed(imgs)
        # patches_embeddings: [B, L, E] 

        # divide to groups
        (embds_with_pos_divided, 
         embds_no_pos_divided, 
         mask) = self.random_divide(patch_embds, num_groups, group_sz, group_duplicates)
        # embds_[with/no]_pos_divided: [B, G, S, E], mask: [B, G, L] (B is just .expand for convenience)

        # apply encoder to groups separately to get reps
        # move groups to batch dim to apply to all separatley in parallel
        reps_divided = self.forward_encoder(embds_with_pos_divided)
        # reps_divided: [B, G, S+1, E] if self.use_cls_token else [B, G, S, E]

        # add mask tokens to each group, and unite group reps for the predictor 'ground truth'
        (reps_divided_expanded, 
         reps_united, 
         reps_contextless_united) = self.arrange_reps(reps_divided, 
                                                                 embds_no_pos_divided, 
                                                                 mask)
        # reps_divided_expanded: [B, G, C+P, E], reps_united: [B, C+P, E]

        # for evaluation
        if mode == 'eval':
            return self.fc_projector(reps_united[:, 0, :])
        elif mode == 'encode_only':
            return reps_united

        # each group tries to predict its own missing reps
        pred = self.forward_predictor(reps_divided_expanded, mask)
        # pred: [B, G, C+P, E]

        # calculate the loss for predicting the missing reps
        (loss, 
        loss_groupwise_invar, 
        loss_batchwise_var, 
        loss_patchwise_var, 
        loss_featurewise_cov, 
        loss_pos_cross_cov) = self.forward_loss(pred, reps_united, mask, reps_contextless_united)
        
        # Evaluate linear probing
        if y is not None:
            loss_lin_prob = self.forward_eval_loss(reps_united[:, 0, :], y)
            loss += loss_lin_prob
        else:
            loss_lin_prob = 0
            
        return loss, loss_groupwise_invar, loss_batchwise_var, loss_patchwise_var, loss_featurewise_cov, loss_pos_cross_cov, loss_lin_prob



def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=1024, decoder_depth=8, decoder_num_heads=16,
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
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 1024 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
