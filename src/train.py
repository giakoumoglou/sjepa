# Copyright (C) 2026. 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.masks.multiblock import MultiblockMaskCollator
from src.masks.random import RandomMaskCollator
from src.masks.block import BlockMaskCollator
from src.masks.grid import GridMaskCollator
from src.masks.rasterized import RasterizedMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets import make_imagenet1k, make_dummy_data

from src.helper import (
    load_checkpoint,
    init_model,
    init_opt)
from src.transforms import make_transforms

# --
log_timings = True
log_freq = 10
checkpoint_freq = 100
# --


def compute_kl_per_dim(predictor, ctx, masks_x, masks, targets):
    """Per-dimension KL(q(z|x,y) || p(z|x)), [B, latent_dim], raw (no
    free-bits clamp). Re-derives mu_p/logvar_p/mu_q/logvar_q using only the
    predictor's public submodules (.predictor_embed, .predictor_pos_embed,
    .prior, .posterior) — mirrors StochasticVisionTransformerPredictor.
    forward() up through the KL computation without needing it to return
    the per-dimension tensor itself. Used by the periodic active-units
    training diagnostic below (kl_diag_freq) and by
    scripts/bench_active_units*.py.
    """
    if not isinstance(masks_x, list):
        masks_x = [masks_x]
    if not isinstance(masks, list):
        masks = [masks]

    B = len(ctx) // len(masks_x)

    x = predictor.predictor_embed(ctx)
    x_pos_embed = predictor.predictor_pos_embed.repeat(B, 1, 1)
    x = x + apply_masks(x_pos_embed, masks_x)
    _, N_ctxt, D = x.shape

    ctx_summary = x.view(B, len(masks_x), N_ctxt, D).mean(dim=(1, 2))
    mu_p, logvar_p = predictor.prior(ctx_summary).chunk(2, dim=-1)

    tgt_summary = apply_masks(targets, [masks[0]]).mean(1)
    post_in = torch.cat([ctx_summary, tgt_summary], dim=-1)
    mu_q, logvar_q = predictor.posterior(post_in).chunk(2, dim=-1)

    kl_per_dim = 0.5 * (
        logvar_p - logvar_q
        + (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp()
        - 1.0
    )
    return kl_per_dim


_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    # -- stochastic predictor
    stochastic = args['meta'].get('stochastic', False)
    latent_dim = args['meta'].get('latent_dim', 128)
    kl_beta_final = args['meta'].get('kl_beta_final', 0.010)
    kl_beta_warmup_frac = args['meta'].get('kl_beta_warmup_frac', 0.3)
    kl_free_bits = float(args['meta'].get('kl_free_bits', 0.0))
    # -- stochastic predictor mechanism-isolation ablations (all default to
    # standard S-JEPA behavior; only meaningful when stochastic=True)
    learn_prior = args['meta'].get('learn_prior', True)
    use_posterior = args['meta'].get('use_posterior', True)
    sample_z = args['meta'].get('sample_z', True)
    use_kl = args['meta'].get('use_kl', True)
    # -- Stochastic Positional Embeddings (Bar et al., ICML 2024), orthogonal
    # to the z-latent mechanism above — perturbs target/masked-token position
    # embeddings, composable with any stochastic= setting.
    use_stop_pos = args['meta'].get('use_stop_pos', False)
    stop_sigma = float(args['meta'].get('stop_sigma', 0.25))
    # -- active-units diagnostic frequency
    kl_diag_freq = args['meta'].get('kl_diag_freq', 0)

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    copy_data = args['data'].get('copy_data', False)
    dummy_data = args['data'].get('dummy_data', False)
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']
    patch_size = args['mask']['patch_size']
    num_enc_masks = args['mask']['num_enc_masks']
    min_keep = args['mask']['min_keep']
    enc_mask_scale = args['mask']['enc_mask_scale']
    num_pred_masks = args['mask']['num_pred_masks']
    pred_mask_scale = args['mask']['pred_mask_scale']
    aspect_ratio = args['mask']['aspect_ratio']
    # -- masking strategy
    mask_strategy = args['mask'].get('strategy', 'multiblock')
    # random params
    random_mask_ratio = args['mask'].get('mask_ratio', 0.6)
    # grid params
    grid_stride = args['mask'].get('grid_stride', 2)

    # -- OPTIMIZATION
    use_ema = args['optimization'].get('use_ema', True)
    # only meaningful when use_ema=False: when True (default), the shared-
    # encoder target branch is still detached (SimSiam-style stop-grad, no
    # separate momentum network but still no double gradient); when False,
    # the target forward pass is fully differentiable too, so nothing but
    # the loss's own structure (e.g. the KL term, for S-JEPA) can prevent
    # collapse. Ignored when use_ema=True (target_encoder params already
    # have requires_grad=False there).
    use_stopgrad = args['optimization'].get('use_stopgrad', True)
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    os.makedirs(folder, exist_ok=True)

    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'initialized rank={rank} world_size={world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))
    kl_logger = None
    if stochastic:
        kl_logger = CSVLogger(
            os.path.join(folder, f'{tag}_kl_r{rank}.csv'),
            ('%d', 'epoch'),
            ('%.6f', 'kl-mean'),
        )
    kl_dim_logger = None
    if stochastic and kl_diag_freq > 0:
        kl_dim_logger = CSVLogger(
            os.path.join(folder, f'{tag}_kl_per_dim_r{rank}.csv'),
            ('%d', 'epoch'),
            *[('%.6f', f'lat{i}') for i in range(latent_dim)],
        )

    # -- init model
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
        stochastic=stochastic,
        latent_dim=latent_dim,
        kl_free_bits=kl_free_bits,
        learn_prior=learn_prior,
        use_posterior=use_posterior,
        sample_z=sample_z,
        use_kl=use_kl,
        use_stop_pos=use_stop_pos,
        stop_sigma=stop_sigma)
    target_encoder = copy.deepcopy(encoder) if use_ema else None

    # -- log model parameter counts
    def _count_params(m):
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total, trainable
    enc_total, enc_train = _count_params(encoder)
    pred_total, pred_train = _count_params(predictor)
    logger.info(f'encoder: {enc_total/1e6:.2f}M params ({enc_train/1e6:.2f}M trainable)')
    logger.info(f'predictor: {pred_total/1e6:.2f}M params ({pred_train/1e6:.2f}M trainable)')
    logger.info(f'total: {(enc_total+pred_total)/1e6:.2f}M params ({(enc_train+pred_train)/1e6:.2f}M trainable)')

    # -- make data transforms
    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter)

    # -- init data-loaders/samplers
    effective_num_workers = num_workers
    if dummy_data:
        logger.info('using synthetic dummy data')
        unsupervised_data, unsupervised_loader, unsupervised_sampler = make_dummy_data(
            crop_size=crop_size,
            batch_size=batch_size,
            collator=None,
            pin_mem=pin_mem,
            num_workers=0,
            world_size=world_size,
            rank=rank,
            length=1024)
    else:
        unsupervised_data, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,
            collator=None,
            pin_mem=pin_mem,
            training=True,
            num_workers=effective_num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            copy_data=copy_data,
            drop_last=True)

    # -- instantiate mask collator
    if mask_strategy == 'block':
        mask_collator = BlockMaskCollator(
            input_size=crop_size,
            patch_size=patch_size,
            block_scale=pred_mask_scale,
            aspect_ratio=aspect_ratio,
            min_keep=min_keep)
    elif mask_strategy == 'random':
        mask_collator = RandomMaskCollator(
            mask_ratio=random_mask_ratio,
            input_size=crop_size,
            patch_size=patch_size,
            min_keep=min_keep)
    elif mask_strategy == 'grid':
        mask_collator = GridMaskCollator(
            input_size=crop_size,
            patch_size=patch_size,
            stride=grid_stride)
    elif mask_strategy == 'rasterized':
        mask_collator = RasterizedMaskCollator(
            input_size=crop_size,
            patch_size=patch_size)
    else:  # 'multiblock'
        mask_collator = MultiblockMaskCollator(
            input_size=crop_size,
            patch_size=patch_size,
            pred_mask_scale=pred_mask_scale,
            enc_mask_scale=enc_mask_scale,
            aspect_ratio=aspect_ratio,
            nenc=num_enc_masks,
            npred=num_pred_masks,
            allow_overlap=allow_overlap,
            min_keep=min_keep)

    unsupervised_loader.collate_fn = mask_collator

    ipe = len(unsupervised_loader)
    total_steps = num_epochs * ipe
    kl_beta_warmup_steps = int(kl_beta_warmup_frac * total_steps)
    logger.info(f'batch_size: {batch_size} per gpu x {world_size} gpus = {batch_size * world_size} effective')

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    # encoder is invoked twice per iteration (once in forward_target, once
    # in forward_context) when use_ema=False and use_stopgrad=False, since
    # neither wraps the "target" forward pass in torch.no_grad(). DDP's
    # static_graph=True assumes a fixed autograd-graph shape across
    # iterations and hangs on the very first step for this combination
    # (observed: NCCL collective timeout after 2h, zero training iterations
    # completed) -- disable it only for this exact flag combination; every
    # other config keeps the faster static_graph=True path unaffected.
    encoder_static_graph = use_ema or use_stopgrad
    encoder = DistributedDataParallel(encoder, static_graph=encoder_static_graph)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    # unwrapped handle for compute_kl_per_dim, which needs direct submodule
    # access (.predictor_embed, .prior, .posterior) that DDP doesn't proxy
    predictor_raw = predictor.module
    if use_ema:
        target_encoder = DistributedDataParallel(target_encoder)
        for p in target_encoder.parameters():
            p.requires_grad = False

    # -- momentum schedule (only used when use_ema=True)
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1)) if use_ema else None

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            if momentum_scheduler is not None:
                next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict() if use_ema else None,
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('epoch: %d' % (epoch + 1))

        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        sl1_meter = AverageMeter()
        kl_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        kl_dim_due = kl_dim_logger is not None and (epoch + 1) % kl_diag_freq == 0
        kl_dim_sum = torch.zeros(latent_dim, device=device) if kl_dim_due else None
        kl_dim_count = 0

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):

            def load_imgs():
                imgs = udata[0].to(device, non_blocking=True)
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (imgs, masks_1, masks_2)
            imgs, masks_enc, masks_pred = load_imgs()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            # -- KL beta: linearly ramp from 0 to kl_beta_final
            global_step = epoch * ipe + itr
            kl_beta = min(kl_beta_final * global_step / max(kl_beta_warmup_steps, 1),
                          kl_beta_final)

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def forward_target():
                    if use_ema:
                        with torch.no_grad():
                            h = target_encoder(imgs)
                    elif use_stopgrad:
                        with torch.no_grad():
                            h = encoder(imgs)
                    else:
                        h = encoder(imgs)
                    h = F.layer_norm(h, (h.size(-1),))
                    B = len(h)
                    h_for_loss = apply_masks(h, masks_pred)
                    h_for_loss = repeat_interleave_batch(h_for_loss, B, repeat=len(masks_enc))
                    return h, h_for_loss

                def forward_context(h_raw):
                    enc_out = encoder(imgs, masks_enc)
                    preds, kl = predictor(enc_out, masks_enc, masks_pred,
                                          targets=h_raw if stochastic else None)
                    return enc_out, preds, kl

                def loss_fn(preds, h_for_loss, kl):
                    sl1 = F.smooth_l1_loss(preds, h_for_loss)
                    loss = sl1
                    kl_term = 0.
                    if stochastic and kl is not None:
                        kl_term = kl_beta * kl
                        loss = loss + kl_term
                    return AllReduce.apply(loss), sl1, kl_term

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    h_raw, h = forward_target()
                    enc_out, preds, kl = forward_context(h_raw)
                    loss, sl1, kl_term = loss_fn(preds, h, kl)

                    kl_per_dim_batch = None
                    if kl_dim_due:
                        with torch.no_grad():
                            kl_per_dim_batch = compute_kl_per_dim(
                                predictor_raw, enc_out, masks_enc, masks_pred, h_raw
                            ).mean(dim=0)   # [latent_dim], averaged over this batch

                # Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Step 3. Momentum update of target encoder
                _m = None
                if use_ema:
                    with torch.no_grad():
                        _m = next(momentum_scheduler)
                        for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                            param_k.data.mul_(_m).add_((1. - _m) * param_q.detach().data)

                return (float(loss), _new_lr, _new_wd, grad_stats,
                        float(sl1), float(kl_term), _m,
                        kl_per_dim_batch)

            (loss, _new_lr, _new_wd, grad_stats, sl1, kl_val, _m,
             kl_per_dim_batch), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            sl1_meter.update(sl1)
            kl_meter.update(kl_val)
            time_meter.update(etime)
            if kl_dim_due:
                kl_dim_sum += kl_per_dim_batch
                kl_dim_count += 1

            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    mom_str = f'{_m:.4f}' if _m is not None else '---'
                    grad_str = ''
                    if grad_stats is not None:
                        grad_str = (' [grad_stats: %.2e %.2e] (%.2e, %.2e)'
                                    % (grad_stats.first_layer, grad_stats.last_layer,
                                       grad_stats.min, grad_stats.max))
                    logger.info(
                        '[%d, %5d] '
                        '[loss: %.3f] [jepa: %.3f] [kl: %.3e] '
                        '[masks: %.0f %.0f] [lr: %.2e] [wd: %.2e] [mom: %s] [mem: %.2e] '
                        '(%.1f ms)%s'
                        % (epoch + 1, itr,
                        loss_meter.avg, sl1_meter.avg, kl_meter.avg,
                        maskA_meter.avg, maskB_meter.avg, _new_lr, _new_wd, mom_str,
                        torch.cuda.max_memory_allocated() / 1024.**2,
                        time_meter.avg, grad_str))

            log_stats()

            assert not np.isnan(loss), 'loss is nan'

        logger.info('avg_loss: %.3f' % loss_meter.avg)
        if kl_logger is not None and rank == 0:
            logger.info('avg_kl: %.6f' % kl_meter.avg)
            kl_logger.log(epoch + 1, kl_meter.avg)
        if kl_dim_due and rank == 0:
            kl_dim_mean = (kl_dim_sum / kl_dim_count).tolist()
            kl_dim_logger.log(epoch + 1, *kl_dim_mean)
        save_checkpoint(epoch + 1)


if __name__ == "__main__":
    main()
