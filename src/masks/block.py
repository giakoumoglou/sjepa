# Copyright (C) 2026. 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from logging import getLogger

import torch
from torch.utils.data import default_collate

from src.masks.multiblock import MultiblockMaskCollator

logger = getLogger()


class BlockMaskCollator(MultiblockMaskCollator):
    """Single random rectangular block masking (BERT / BEiT style).
    https://arxiv.org/abs/1810.04805 / https://arxiv.org/abs/2106.08254

    One rectangular block is sampled as the prediction target; every other
    patch is context (visible to the encoder).  This is the simplest
    block-based masking strategy:

        Context = all patches NOT in the target block
        Target  = one rectangular block

    Contrast with MultiblockMaskCollator (I-JEPA), which samples a separate
    context block and four target blocks independently.

    Inherits ``_sample_block_size`` and ``_sample_block_mask`` from
    ``MultiblockMaskCollator`` to reuse the block-sampling logic.

    Args:
        input_size: image side length in pixels, int or (H, W).
        patch_size: patch side length in pixels.
        block_scale: (min, max) fraction of total patches for the target block.
        aspect_ratio: (min, max) aspect ratio of the target block.
        min_keep: minimum patches required for a valid block.
    """

    def __init__(
        self,
        input_size=224,
        patch_size: int = 16,
        block_scale=(0.15, 0.50),
        aspect_ratio=(0.75, 1.5),
        min_keep: int = 4,
    ):
        # Pass block_scale as pred_mask_scale; enc_mask_scale / nenc / npred
        # are not used by our __call__ but must satisfy the parent __init__.
        super().__init__(
            input_size=input_size,
            patch_size=patch_size,
            enc_mask_scale=(0.85, 1.0),   # unused — we override __call__
            pred_mask_scale=block_scale,
            aspect_ratio=aspect_ratio,
            nenc=1,
            npred=1,
            min_keep=min_keep,
            allow_overlap=False,
        )
        self.block_scale = block_scale

    def __call__(self, batch):
        """Collate a batch and produce single-block encoder / predictor masks.

        Returns:
            collated_batch: standard default_collate output.
            collated_masks_enc: list with one LongTensor (B, N_ctx) —
                                all patches outside the target block.
            collated_masks_pred: list with one LongTensor (B, N_tgt) —
                                 the target block patches.
        """
        B = len(batch)
        collated_batch = default_collate(batch)

        # Shared block size for the whole batch (same h × w, random position
        # per image — identical convention to MultiblockMaskCollator).
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        b_size = self._sample_block_size(
            generator=g,
            scale=self.block_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )

        ctx_list, tgt_list = [], []
        min_ctx = self.N
        min_tgt = self.N

        for _ in range(B):
            tgt, _ = self._sample_block_mask(b_size)   # (tgt_indices, complement_2d)

            # Context = full complement (not a sub-block like in I-JEPA)
            in_tgt = torch.zeros(self.N, dtype=torch.bool)
            in_tgt[tgt] = True
            ctx = torch.arange(self.N)[~in_tgt]

            ctx_list.append(ctx)
            tgt_list.append(tgt)
            min_ctx = min(min_ctx, ctx.numel())
            min_tgt = min(min_tgt, tgt.numel())

        ctx_tensor = torch.stack([c[:min_ctx] for c in ctx_list])   # (B, N_ctx)
        tgt_tensor = torch.stack([t[:min_tgt] for t in tgt_list])   # (B, N_tgt)

        return collated_batch, [ctx_tensor], [tgt_tensor]
