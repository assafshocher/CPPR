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
                 w_batchwise_loss=1., w_patchwise_loss=0., w_batchwise_cls_loss=0.):
        super().__init__()

        # --------------------------------------------------------------------------
        self.use_cls_token = use_cls_token
        self.slim_predictor = slim_predictor
        self.cls_predict_loss = cls_predict_loss
        self.debug_mode = debug_mode
        self.temperature = temperature
        self.w_batchwise_loss = w_batchwise_loss
        self.w_patchwise_loss = w_patchwise_loss
        self.w_batchwise_cls_loss = w_batchwise_cls_loss
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        if not debug_mode:
            self.blocks = nn.ModuleList([
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(depth)])
            self.norm = norm_layer(embed_dim)
            # self.batch_norm = nn.SyncBatchNorm(embed_dim, affine=False)
        # --------------------------------------------------------------------------
        if cls_predict_loss:
            self.cls_predict_tokens_mlp = nn.Sequential(nn.Linear(num_patches, embed_dim, bias=True),
                                                        nn.GELU(), 
                                                        nn.Linear(embed_dim, embed_dim, bias=True))

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        if not debug_mode:
            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(decoder_depth)])

            self.decoder_norm = norm_layer(decoder_embed_dim)
            self.decoder_pred = nn.Linear(decoder_embed_dim, embed_dim, bias=True) # decoder to embedding
        # --------------------------------------------------------------------------

        self.fc_projector = torch.nn.Linear(embed_dim, 1000)
        self.fc_projector = torch.nn.Sequential(torch.nn.BatchNorm1d(embed_dim, affine=False), self.fc_projector)
        self.cross_entropy = torch.nn.CrossEntropyLoss()

        if not self.debug_mode:
            self.initialize_weights()
    
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
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


    def patchify(self, imgs):
        """
        imgs: (B, 3, H, W)
        x: (B, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    
    def patch_embed_and_add_pos(self, imgs):
        if self.debug_mode:
            return self.patchify(imgs)

        # embed patches
        x = self.patch_embed(imgs)

        # add pos embed w/o cls token
        x = x + self.pos_embed

        return x

        
    def random_divide(self, x, num_groups, group_sz, group_duplicates):
        """
        x: [B*G, L, E] if group_duplicates else [B, L, E]
        x_out: [B, G, S, E]
        mask: [B, G, L], indicator along G-dim which group this patch belongs to, all 0 means no-one.
        """
        A, L, E = x.shape  # A might be B*G or just B
        G = num_groups
        S = group_sz

        if group_duplicates:
            B = A // G
            x = x.view(B, G, L, E)
        else:
            B = A
            x = x.unsqueeze(1).expand(B, G, L, E)
       
        assert G * S <= L

        perm = torch.randperm(L, device=x.device)[None, :]
        lower_bds = S * torch.arange(G, device=x.device)[:, None]
        mask = (perm >= lower_bds) * (perm < lower_bds + S)[None, ...].expand(B, G, L)
        x = x[mask]
        
        return x.view(B, G, S, E), mask

    def forward_encoder(self, x):
        """
        patches_embeddings_divided: [B*G, S, E]
        """

        # append cls token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)  # x: [B*G, 1+S, E]

        # apply Transformer blocks
        if not self.debug_mode:
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)

        return x

        
    def arrange_representations(self, representations_divided, mask):
        """
        representations_divided: [B, G, S+1, E] if self.use_cls_token else [B, G, S, E]
        mask: [B, G, L], indicator along G-dim which group this patch belongs to, all 0 means no-one.
        """
        B, G, S, E = representations_divided.shape  # if cls token used then this S is wrong, will be fixed below 
        L = mask.shape[-1]
        
        # take care of the cls token
        if self.use_cls_token:
            S = S - 1  # fix S because we intialy added the cls token to the count
            orig_cls_token = representations_divided[:, :, 0, :]  # [B, G, E]
            representations_divided = representations_divided[:, :, 1:, :]

            # create sort of masked cls tokens for predicting other clss, when using this option
            if self.cls_predict_loss:
                if not self.debug_mode:
                    # the masked clss are a function of the mask
                    cls_tokens = self.cls_predict_tokens_mlp(mask * 1.).unsqueeze(1).float()  # [B, 1, G, E]
                else:
                    # this is for debugging
                    cls_tokens = torch.randint(10, (B, 1, G, E), device=orig_cls_token.device)
                    orig_cls_token = torch.randint(10, (B, G, E), device=orig_cls_token.device)

                cls_tokens = cls_tokens.repeat(1, G, 1, 1)  # [B, G, G, E]
                
                # for every group, its own cls prediction token is just its own cls token 
                cls_tokens[:, range(G), range(G), :] = orig_cls_token[:, range(G), :]
            else:
                cls_tokens = orig_cls_token.unsqueeze(2)

        representations_divided_expanded = self.mask_token.repeat(B, G, L, 1)
        representations_divided_expanded[mask] = representations_divided.reshape(-1, E)  # [B, G, L, E]
        
        # unite the divided representations by summing along G (only one element is not multiplied by 0)
        representations_united = (representations_divided_expanded * mask.unsqueeze(-1)).sum(1)  # [B, L, E]
        # in this config we throw away mask tokens for patches that are not in any group
        if self.slim_predictor:
            global_mask = mask[0].bool().any(0)  # [L]
            representations_divided_expanded = representations_divided_expanded[:, :, global_mask, :]  # [B, G, S, E]
            representations_united = representations_united[:, global_mask, :]  # [B, S, E]

        # add the mask tokens in the beginning (adds 0 or G more tokens)
        if self.use_cls_token:
            representations_divided_expanded = torch.cat([cls_tokens, representations_divided_expanded],
                                                         2)  # [B, G, G+P, E]
            if not self.cls_predict_loss:
                orig_cls_token = orig_cls_token.unsqueeze(2).mean(1)
            representations_united = torch.cat([orig_cls_token, representations_united], 1)  # [B, 1+P, E]


        return representations_divided_expanded, representations_united


    def forward_predictor(self, representations_divided, mask):
        """
        representations_divided: [B, G, C+P, E]
        """
        B, G, C_plus_P, E = representations_divided.shape
        C = (G - 1) * self.cls_predict_loss + self.use_cls_token
        P = C_plus_P - C

        # move groups to batch dim to apply blocks
        x = representations_divided.view(B*G, C+P, E)

        if not self.debug_mode:
            # embed tokens
            x = self.decoder_embed(x)

            # add pos embed (need to slim it)
            L = self.decoder_pos_embed.shape[1]
            indices = mask.sum(1, keepdim=True).expand(B, G, L).reshape(B*G, L)
            pos_embed = self.decoder_pos_embed.expand(B*G, -1, -1)
            x[:, C:, :] = x[:, C:, :] + pos_embed[indices.bool()].view(B*G, P, -1)  # note E is encoder embed sz so we used -1 here.

            # apply Transformer blocks
            for blk in self.decoder_blocks:
                x = blk(x)
            x = self.decoder_norm(x)

            # predictor projection
            x = self.decoder_pred(x)

        pred = x.view(B, G, C+P, E)

        return pred


    def forward_loss(self, pred, rep, mask):
        """
        pred: [B, G, C+P, E]
        rep: [B, C+P, E]
        mask: [B, G, L]
        """
        B, G, C_plus_P, E = pred.shape
        C = (G - 1) * self.cls_predict_loss + self.use_cls_token
        P = C_plus_P - C
        L = mask.shape[-1]

        # # normalize before dot prod. equivalent to cosine-sim
        pred = F.normalize(pred, dim=-1)
        rep = F.normalize(rep, dim=-1) # .detach()
        # separate the cls tokens (cls will be empty if not using because C==0)
        pred_cls, pred = pred[:, :, :C, :], pred[:, :, C:, :]
        rep_cls, rep = rep[:, :C, :], rep[:, C:, :]

        # batchwise is similarity between pred to all the representations of same location in the batch
        # patchwise is  similarity between pred and all the representations of other patches in the same image
        batchwise_similarity_matrix = torch.einsum('bgpe,cpe->gpbc', pred, rep)
        # pred = rearrange(pred, 'B G P E -> G P B 1 E')
        # rep = rearrange(rep, 'B P E -> 1 P 1 B E')
        # batchwise_similarity_matrix = (pred - rep).pow(2).sum(-1)  # [G P B B]
        
        
        if self.w_patchwise_loss:
            patchwise_similarity_matrix = torch.einsum('bgpe,bqe->bgpq', pred, rep)
        if self.cls_predict_loss:  # C==G
            batchwise_cls_similarity_matrix = torch.einsum('bgce,dce->gcbd', pred_cls, rep_cls)

        # for CE we create logits, s.t. the number of classes is the number of exapmples we contrast with
        batchwise_logits = batchwise_similarity_matrix.reshape(G*P*B, B)  / self.temperature
        if self.w_patchwise_loss:
            patchwise_logits = patchwise_similarity_matrix.reshape(B*G*P, P)  / self.temperature
        if self.cls_predict_loss:
            batchwise_cls_logits = batchwise_cls_similarity_matrix.reshape(G*C*B, B) / self.temperature

        # since the positive examples are on the main diag, the labels are 0,1,2,3...
        batchwise_labels = torch.arange(B, dtype=torch.long, device=pred.device).repeat(G*P)
        if self.w_patchwise_loss:
            patchwise_labels = torch.arange(P, dtype=torch.long, device=pred.device).repeat(B*G)
        if self.cls_predict_loss:
            batchwise_cls_labels = torch.arange(B, dtype=torch.long, device=pred.device).repeat(G*C)

        # apply CE loss. we don't reduce so we can mask out self-predictions and weight cls
        batchwise_loss = self.criterion(batchwise_logits, batchwise_labels).view(G, P, B)
        if self.w_patchwise_loss:
            patchwise_loss = self.criterion(patchwise_logits, patchwise_labels).view(B, G, P)
        if self.cls_predict_loss:
            batchwise_cls_loss = self.criterion(batchwise_cls_logits, batchwise_cls_labels).view(G, C, B)

        # reshape losses to natural shapes
        batchwise_loss = rearrange(batchwise_loss, 'G P B -> B G P')
        if self.w_patchwise_loss:
            patchwise_loss = rearrange(patchwise_loss, 'B G P -> B G P')
        if self.cls_predict_loss:    # C==G
            batchwise_cls_loss = rearrange(batchwise_cls_loss, 'G C B -> B G C')

        # mask out the already known values and average into a single scalar per loss
        if self.slim_predictor:
            mask = mask[mask.bool().any(1, keepdim=True).expand(B, G, L)].view(B, G, P)
        batchwise_loss = (batchwise_loss * (~mask)).sum() / (~mask).sum()
        
        if self.w_patchwise_loss:
            patchwise_loss = (patchwise_loss * (~mask)).sum() / (~mask).sum()
        else:
            patchwise_loss = torch.zeros(1, device=pred.device)
        
        if self.cls_predict_loss:
            cls_mask = torch.eye(G, device=pred.device)[None, ...]
            batchwise_cls_loss = (batchwise_cls_loss * (1-cls_mask)).sum() / (G*(G-1)*B)
        else:
            batchwise_cls_loss = torch.zeros(1, device=pred.device)
        
        # combine all the losses
        loss = (self.w_batchwise_loss * batchwise_loss + 
                self.w_patchwise_loss * patchwise_loss +
                self.w_batchwise_cls_loss * batchwise_cls_loss)

        if self.debug_mode:
            return (loss, batchwise_loss, patchwise_loss, batchwise_cls_loss,
                    (batchwise_similarity_matrix, patchwise_similarity_matrix, batchwise_cls_similarity_matrix,
                    batchwise_logits, patchwise_logits, batchwise_cls_logits, 
                    batchwise_labels, patchwise_labels, batchwise_cls_labels))

        return loss, batchwise_loss, patchwise_loss, batchwise_cls_loss, None

    def forward_eval_loss(self, h, y):
        h = h.detach()
        y_h_pred = self.fc_projector(h)
        loss_y_h = self.cross_entropy(y_h_pred, y)
        return loss_y_h

    def forward(self, imgs, num_groups, group_sz, y=None, encode_only=False, group_duplicates=True, mode='train'):
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

        # first processing of the images to embedded patches with pos embeddings
        patches_embeddings = self.patch_embed_and_add_pos(imgs)
        # patches_embeddings: [B, L, E] 

        # divide to groups
        patches_embeddings_divided, mask = self.random_divide(patches_embeddings, num_groups, group_sz,
                                                              group_duplicates)
        # patches_embeddings_divided: [B, G, S, E], mask: [B, G, L] (B is just .expand for convenience)

        # apply encoder to groups separately to get representations
        # move groups to batch dim to apply to all separatley in parallel
        representations_divided = self.forward_encoder(patches_embeddings_divided.flatten(0, 1))
        B, G, S, E = patches_embeddings_divided.shape
        representations_divided = representations_divided.view(B, G, S + self.use_cls_token, E)
        # representations_divided: [B, G, S+1, E] if self.use_cls_token else [B, G, S, E]

        # add mask tokens to each group, and unite group reps for the predictor 'ground truth'
        (representations_divided_expanded, representations_united) = self.arrange_representations(
            representations_divided, mask)
        # representations_divided_expanded: [B, G, C+P, E], representations_united: [B, C+P, E]

        if mode == 'eval':
            return self.fc_projector(representations_united[:, 0, :])

        # for evaluation etc. return the representations here.
        if encode_only:
            return representations_united  # [B, C+P, E]

        # each group tries to predict its own missing representations
        pred = self.forward_predictor(representations_divided_expanded, mask)
        # pred: [B, G, C+P, E]

        # calculate the loss for predicting the missing representations
        loss, loss_batchwise, loss_patchwise, loss_cls, other_stats = self.forward_loss(pred, representations_united,
                                                                                        mask)
        if y is not None:
            loss += self.forward_eval_loss(representations_united[:, 0, :], y)

        # return all intermidates when debugging
        if self.debug_mode:
            return (patches_embeddings, patches_embeddings_divided, mask, representations_divided,
                    representations_divided_expanded, representations_united, pred, loss, loss_batchwise, 
                    loss_patchwise, loss_cls, *other_stats)
            
        return loss, pred, mask, loss_batchwise, loss_patchwise, loss_cls






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
