# Copyright (C) 2026. 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from functools import partial
import numpy as np

import torch
import torch.nn as nn

from src.utils.tensors import (
    trunc_normal_,
    repeat_interleave_batch
)
from src.masks.utils import apply_masks


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid length
    return:
    pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ConvEmbed(nn.Module):
    """
    3x3 Convolution stems for ViT following ViTC models
    """

    def __init__(self, channels, strides, img_size=224, in_chans=3, batch_norm=True):
        super().__init__()
        # Build the stems
        stem = []
        channels = [in_chans] + channels
        for i in range(len(channels) - 2):
            stem += [nn.Conv2d(channels[i], channels[i+1], kernel_size=3,
                               stride=strides[i], padding=1, bias=(not batch_norm))]
            if batch_norm:
                stem += [nn.BatchNorm2d(channels[i+1])]
            stem += [nn.ReLU(inplace=True)]
        stem += [nn.Conv2d(channels[-2], channels[-1], kernel_size=1, stride=strides[-1])]
        self.stem = nn.Sequential(*stem)

        # Comptute the number of patches
        stride_prod = int(np.prod(strides))
        self.num_patches = (img_size[0] // stride_prod)**2

    def forward(self, x):
        p = self.stem(x)
        return p.flatten(2).transpose(1, 2)


class VisionTransformerPredictor(nn.Module):
    """ Vision Transformer Predictor (deterministic) """
    def __init__(
        self,
        num_patches,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        use_stop_pos=False,
        stop_sigma=0.25,
        **kwargs
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim),
                                                requires_grad=False)
        predictor_pos_embed = get_2d_sincos_pos_embed(self.predictor_pos_embed.shape[-1],
                                                      int(num_patches**.5),
                                                      cls_token=False)
        self.predictor_pos_embed.data.copy_(torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

        # ── Stochastic Positional Embeddings (StoP; Bar et al., ICML 2024) ──
        # Perturbs target/masked-token positional embeddings with learned-
        # projection Gaussian noise: psi_hat = stop_proj(n) + psi, n ~
        # N(0, stop_sigma^2 I). Orthogonal to the z-latent mechanism in the
        # stochastic subclass — lives here so it works for either predictor.
        # NOTE: documented approximation, not byte-exact — the paper ties
        # its noise-projection matrix to a context-projection matrix from
        # their own architecture; this repo has no directly analogous single
        # matrix to tie to, so stop_proj is its own small learned layer
        # instead. Train-time only (gated on self.training); at inference/
        # eval the positional embedding is left deterministic.
        self.use_stop_pos = use_stop_pos
        self.stop_sigma = stop_sigma
        if self.use_stop_pos:
            self.stop_proj = nn.Linear(predictor_embed_dim, predictor_embed_dim)
            trunc_normal_(self.stop_proj.weight, std=self.init_std)
            nn.init.constant_(self.stop_proj.bias, 0)
        else:
            self.stop_proj = None

    def _apply_stop_pos(self, pos_embs):
        if self.use_stop_pos and self.training:
            noise = torch.randn_like(pos_embs) * self.stop_sigma
            pos_embs = pos_embs + self.stop_proj(noise)
        return pos_embs

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks_x, masks, targets=None):
        """
        Args:
            x:       context tokens  [B*nenc, N_ctx, encoder_dim]
            masks_x: list of context masks
            masks:   list of target masks
            targets: unused in deterministic predictor (accepted for API compatibility)
        Returns:
            predictions [B*nenc*npred, N_pred, encoder_dim], kl=None
        """
        del targets  # not used by deterministic predictor; accepted for API parity with stochastic
        assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'

        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks, list):
            masks = [masks]

        B = len(x) // len(masks_x)

        x = self.predictor_embed(x)

        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += apply_masks(x_pos_embed, masks_x)

        _, N_ctxt, D = x.shape

        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        pos_embs = self._apply_stop_pos(pos_embs)
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        pred_tokens += pos_embs

        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        x = x[:, N_ctxt:]
        return self.predictor_proj(x), None


class StochasticVisionTransformerPredictor(VisionTransformerPredictor):
    """
    VA-JEPA predictor: extends VisionTransformerPredictor with a
    conditional VAE latent variable z.

    During training  (targets provided):
      z ~ q(z | context, target)   via reparameterisation
      loss += beta * KL(q || p)

    During inference (targets=None):
      z  = mu_prior(context)        deterministic

    Mechanism-isolation flags (all default True = standard S-JEPA):
      learn_prior:    if False, p(z|context) is a fixed N(0,I) instead of a
                       learned context-conditioned MLP (no self.prior module).
      use_posterior:  if False, no q(z|context,target) network exists (no
                       self.posterior module) — z always comes from the prior.
      sample_z:       if False, z is the distribution mean, no reparameterized
                       sampling noise, during training.
      use_kl:         if False, no KL term is ever computed/returned.
    """

    def __init__(
        self,
        num_patches,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        latent_dim=128,
        kl_free_bits=0.0,
        learn_prior=True,
        use_posterior=True,
        sample_z=True,
        use_kl=True,
        **kwargs
    ):
        super().__init__(
            num_patches=num_patches,
            embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=depth,
            num_heads=num_heads,
            **kwargs)
        self.latent_dim = latent_dim
        self.kl_free_bits = kl_free_bits
        self.learn_prior = learn_prior
        self.use_posterior = use_posterior
        self.sample_z = sample_z
        self.use_kl = use_kl

        # p(z | context): 2-layer MLP on mean-pooled context tokens
        if self.learn_prior:
            self.prior = nn.Sequential(
                nn.Linear(predictor_embed_dim, predictor_embed_dim // 2),
                nn.GELU(),
                nn.Linear(predictor_embed_dim // 2, 2 * latent_dim),
            )
        else:
            self.prior = None   # fixed N(0, I); mu_p=0, logvar_p=0 at forward time

        # q(z | context, target): 2-layer MLP on context + target summaries
        if self.use_posterior:
            self.posterior = nn.Sequential(
                nn.Linear(predictor_embed_dim + embed_dim, predictor_embed_dim),
                nn.GELU(),
                nn.Linear(predictor_embed_dim, 2 * latent_dim),
            )
        else:
            self.posterior = None

        # project z into predictor embedding space (injected as prefix token)
        self.z_proj = nn.Linear(latent_dim, predictor_embed_dim)

        # initialise new parameters
        for m in [self.prior, self.posterior, self.z_proj]:
            if m is None:
                continue
            for layer in (m.modules() if hasattr(m, 'modules') else [m]):
                if isinstance(layer, nn.Linear):
                    trunc_normal_(layer.weight, std=self.init_std)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def forward(self, x, masks_x, masks, targets=None):
        """
        Args:
            x:       context tokens  [B*nenc, N_ctx, encoder_dim]
            masks_x: list of context masks
            masks:   list of target masks
            targets: full target encoder output [B, N_all, encoder_dim] — only
                     provided during training so the posterior can be computed.
        Returns:
            predictions  [B*nenc*npred, N_pred, encoder_dim]
            kl           scalar KL loss (None at inference)
        """
        assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'

        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks, list):
            masks = [masks]

        B = len(x) // len(masks_x)

        # ── project encoder dim → predictor dim ────────────────────────────
        x = self.predictor_embed(x)          # [B*nenc, N_ctx, D]

        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += apply_masks(x_pos_embed, masks_x)

        _, N_ctxt, D = x.shape

        # ── stochastic latent z ─────────────────────────────────────────────
        # ctx_summary: average over context tokens and encoder-mask repetitions
        ctx_summary = x.view(B, len(masks_x), N_ctxt, D).mean(dim=(1, 2))   # [B, D]

        if self.learn_prior:
            mu_p, logvar_p = self.prior(ctx_summary).chunk(2, dim=-1)   # [B, latent_dim]
        else:
            mu_p = torch.zeros(B, self.latent_dim, device=x.device, dtype=ctx_summary.dtype)
            logvar_p = torch.zeros_like(mu_p)   # fixed N(0, I)

        kl = None
        if self.use_posterior and targets is not None:
            # targets: [B, N_all, encoder_dim]
            # summarise over the first prediction-mask region
            # NOTE: masks[0] must be wrapped in a list — apply_masks expects
            # a list of mask tensors and iterates over it; passing a raw
            # (B, N_pred) tensor made it iterate over the batch dimension
            # instead, so every image's posterior was conditioned on image
            # 0's target representation instead of its own.
            tgt_summary = apply_masks(targets, [masks[0]]).mean(1)           # [B, encoder_dim]
            post_in = torch.cat([ctx_summary, tgt_summary], dim=-1)
            mu_q, logvar_q = self.posterior(post_in).chunk(2, dim=-1)

            if self.sample_z:
                # reparameterisation trick
                std_q = (0.5 * logvar_q).exp()
                z = mu_q + std_q * torch.randn_like(std_q)
            else:
                z = mu_q   # deterministic posterior mean, no sampling noise

            if self.use_kl:
                # analytical KL(q || p) for diagonal Gaussians — reduces to the
                # standard closed-form KL(q || N(0,I)) when learn_prior=False
                # (mu_p=0, logvar_p=0)
                kl_per_dim = 0.5 * (
                    logvar_p - logvar_q
                    + (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp()
                    - 1.0
                )  # [B, latent_dim]
                if self.kl_free_bits > 0:
                    kl_per_dim = kl_per_dim.clamp(min=self.kl_free_bits)
                kl = kl_per_dim.sum(-1).mean()
        elif targets is not None and self.sample_z:
            # no posterior (use_posterior=False): sample straight from the
            # prior every step, train and eval alike — e.g. pure-noise-
            # injection ablation, where the prior is a fixed N(0,I).
            std_p = (0.5 * logvar_p).exp()
            z = mu_p + std_p * torch.randn_like(std_p)
        else:
            # inference (targets=None), or use_posterior=False+sample_z=False
            # (e.g. matched-capacity deterministic-prefix-token ablation):
            # deterministic prior mean.
            z = mu_p

        # inject z as a prefix token (broadcast over nenc repetitions)
        z_token = self.z_proj(z).unsqueeze(1)                               # [B, 1, D]
        z_token = z_token.repeat_interleave(len(masks_x), dim=0)            # [B*nenc, 1, D]
        x = torch.cat([z_token, x], dim=1)                                  # [B*nenc, 1+N_ctxt, D]
        N_ctxt_total = N_ctxt + 1

        # ── build mask tokens (unchanged from deterministic predictor) ──────
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        pos_embs = self._apply_stop_pos(pos_embs)
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        pred_tokens += pos_embs

        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # strip z_token + context tokens; keep only target predictions
        x = x[:, N_ctxt_total:]
        return self.predictor_proj(x), kl


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=12,
        predictor_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        # --
        self.patch_embed = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # --
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                            int(self.patch_embed.num_patches**.5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # ------
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks=None, return_layers=None):
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        if return_layers is not None:
            depth = len(self.blocks)
            invalid = [l for l in return_layers if l < 1 or l > depth]
            if invalid:
                raise ValueError(f'return_layers {invalid} out of range [1, {depth}]')

        # -- patchify x
        x = self.patch_embed(x)
        B, N, D = x.shape

        # -- add positional embedding to x
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed

        # -- mask x
        if masks is not None:
            x = apply_masks(x, masks)

        # -- fwd prop
        layer_set = set(return_layers) if return_layers is not None else None
        intermediates = {}
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if layer_set is not None and (i + 1) in layer_set:
                intermediates[i + 1] = x

        if return_layers is not None:
            return intermediates

        if self.norm is not None:
            x = self.norm(x)

        return x

    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N:
            return pos_embed
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)


def vit_predictor(**kwargs):
    model = VisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def vit_stochastic_predictor(**kwargs):
    model = StochasticVisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_giant(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


VIT_EMBED_DIMS = {
    'vit_tiny': 192,
    'vit_small': 384,
    'vit_base': 768,
    'vit_large': 1024,
    'vit_huge': 1280,
    'vit_giant': 1408,
}


if __name__ == '__main__':
    B = 2
    IMG_SIZE = 224
    PATCH = 16
    N = (IMG_SIZE // PATCH) ** 2   # 196

    x = torch.randn(B, 3, IMG_SIZE, IMG_SIZE)
    # context: first 150 patches; target: remaining 46 patches
    masks_enc  = [torch.arange(150).unsqueeze(0).expand(B, -1)]
    masks_pred = [torch.arange(150, N).unsqueeze(0).expand(B, -1)]

    configs = [
        ('vit_tiny',  vit_tiny,  192, 3),
        ('vit_small', vit_small, 384, 6),
        ('vit_base',  vit_base,  768, 12),
    ]

    for name, factory, embed_dim, num_heads in configs:
        print(f'\n--- {name} ---')
        encoder = factory(patch_size=PATCH, img_size=[IMG_SIZE])
        encoder.eval()

        # 1. Full forward (no masking)
        with torch.no_grad():
            out = encoder(x)
        print(f'  full forward   : {tuple(x.shape)} -> {tuple(out.shape)}')

        # 2. Masked forward (context only)
        with torch.no_grad():
            out_ctx = encoder(x, masks=masks_enc)
        print(f'  masked forward : {tuple(x.shape)}, mask {tuple(masks_enc[0].shape)} -> {tuple(out_ctx.shape)}')

        # 3. return_layers (hierarchical)
        with torch.no_grad():
            intermediates = encoder(x, return_layers=[4, 8, 12])
        for layer_idx, feat in intermediates.items():
            print(f'  layer {layer_idx:>2d}        : {tuple(feat.shape)}')

        # 4. Predictor (single head)
        pred_emb = embed_dim // 2
        predictor = vit_predictor(
            num_patches=N, embed_dim=embed_dim,
            predictor_embed_dim=pred_emb, depth=6, num_heads=num_heads)
        predictor.eval()
        with torch.no_grad():
            z = predictor(out_ctx, masks_enc, masks_pred)
        print(f'  predictor      : {tuple(out_ctx.shape)} -> {tuple(z.shape)}')

        # 5. Predictor (multi-head, K=3)
        predictor_mh = vit_predictor(
            num_patches=N, embed_dim=embed_dim,
            predictor_embed_dim=pred_emb, depth=6, num_heads=num_heads,
            num_pred_heads=3)
        predictor_mh.eval()
        with torch.no_grad():
            zs = predictor_mh(out_ctx, masks_enc, masks_pred)
        for k, z_k in enumerate(zs):
            print(f'  predictor head {k}: {tuple(z_k.shape)}')

    print('\nAll VisionTransformer tests passed.')
