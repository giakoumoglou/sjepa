# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys

import torch

import src.models.vision_transformer as vit
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)
from src.utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_checkpoint(
    device,
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        pretrained_dict = checkpoint['encoder']
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        pretrained_dict = checkpoint['predictor']
        msg = predictor.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained predictor from epoch {epoch} with msg: {msg}')

        if target_encoder is not None:
            pretrained_dict = checkpoint['target_encoder']
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(f'loaded pretrained target_encoder from epoch {epoch} with msg: {msg}')

        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read_path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'encountered exception when loading checkpoint {e}')
        epoch = 0

    return encoder, predictor, target_encoder, opt, scaler, epoch


def init_model(
    device,
    patch_size=16,
    model_name='vit_base',
    crop_size=224,
    pred_depth=6,
    pred_emb_dim=384,
    stochastic=False,
    latent_dim=128,
    kl_free_bits=0.0,
    learn_prior=True,
    use_posterior=True,
    sample_z=True,
    use_kl=True,
    use_stop_pos=False,
    stop_sigma=0.25,
):
    """Instantiate encoder + predictor pair.

    Args:
        stochastic: if True, use the S-JEPA stochastic predictor (CVAE latent z).
        latent_dim: dimensionality of the latent variable z (stochastic only).
        kl_free_bits: minimum KL per dimension (free bits); 0 disables.
        learn_prior: if False, p(z|context) is fixed N(0,I) instead of learned.
        use_posterior: if False, no q(z|context,target) network exists; z is
            always drawn from the prior (context-only, or fixed N(0,I)).
        sample_z: if False, z is the distribution mean (no reparameterized
            sampling noise) during training.
        use_kl: if False, no KL term is computed/returned regardless of
            whether a posterior exists.
        use_stop_pos: Stochastic Positional Embeddings (Bar et al., ICML
            2024) — perturbs target/masked-token positional embeddings with
            learned-projection Gaussian noise. Orthogonal to the z-latent
            mechanism; composable with stochastic=True or False.
        stop_sigma: std of the StoP noise (paper default 0.25).
    """
    stoch_kwargs = {
        'latent_dim': latent_dim, 'kl_free_bits': kl_free_bits,
        'learn_prior': learn_prior, 'use_posterior': use_posterior,
        'sample_z': sample_z, 'use_kl': use_kl,
    } if stochastic else {}
    stoch_kwargs['use_stop_pos'] = use_stop_pos
    stoch_kwargs['stop_sigma'] = stop_sigma
    encoder = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size)
    pred_fn = 'vit_stochastic_predictor' if stochastic else 'vit_predictor'
    predictor = vit.__dict__[pred_fn](
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads,
        **stoch_kwargs)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    return encoder, predictor


def init_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25,
):
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    logger.info('using adamw')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch))
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler
