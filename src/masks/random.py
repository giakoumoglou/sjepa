# Copyright (C) 2026. 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from typing import Tuple

import torch
from torch.utils.data import default_collate

from src.masks.base import BaseMaskCollator

logger = getLogger()


class RandomMaskCollator(BaseMaskCollator):
    """MAE: Masked Autoencoders Are Scalable Vision Learners
    https://arxiv.org/abs/2111.06377

    Randomly partitions the N patches into context (kept) and target (masked).
    The mask ratio is sampled uniformly from [mask_ratio[0], mask_ratio[1]]
    once per batch using a seeded generator, giving all images in the batch the
    same number of context / target patches.

    Subclasses can override ``_sample_masks`` to replace the uniform random
    selection with any other strategy (e.g. cluster-based).

    Args:
        mask_ratio: fixed float (e.g. 0.6) or (min, max) range sampled each batch.
        input_size: image side length in pixels, int or (H, W).
        patch_size: patch side length in pixels.
        min_keep: minimum number of context patches to retain.
    """

    def __init__(
        self,
        mask_ratio=0.6,
        input_size=224,
        patch_size: int = 16,
        min_keep: int = 4,
    ):
        super().__init__(input_size=input_size, patch_size=patch_size)
        # Accept either a fixed float or a (min, max) range
        if isinstance(mask_ratio, (int, float)):
            mask_ratio = (float(mask_ratio), float(mask_ratio))
        self.mask_ratio = mask_ratio
        self.min_keep = min_keep
        # Set by __call__ before the per-image loop; available to _sample_masks
        self._num_keep: int = 0

    def _sample_masks(
        self,
        img: torch.Tensor,
        g: torch.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (ctx_indices, tgt_indices) for a single image.

        Override in subclasses to change target-selection strategy.
        ``self._num_keep`` holds the batch-level context size.

        Args:
            img: image tensor (C, H, W) — available for feature-based methods.
            g: per-image seeded generator for reproducibility.

        Returns:
            ctx_indices: 1-D LongTensor of context (encoder-visible) patch indices.
            tgt_indices: 1-D LongTensor of target (to-be-predicted) patch indices.
        """
        perm = torch.randperm(self.N, generator=g)
        return perm[:self._num_keep], perm[self._num_keep:]

    def __call__(self, batch):
        """Collate a batch and produce random encoder / predictor masks.

        Returns:
            collated_batch: standard default_collate output.
            collated_masks_enc: list with one LongTensor (B, N_ctx).
            collated_masks_pred: list with one LongTensor (B, N_tgt).
        """
        B = len(batch)
        collated_batch = default_collate(batch)
        imgs: torch.Tensor = collated_batch[0]   # (B, C, H, W)

        seed = self.step()
        # Batch-level ratio: same for all images in this batch
        g_batch = torch.Generator()
        g_batch.manual_seed(seed)
        lo, hi = self.mask_ratio
        ratio = lo + torch.rand(1, generator=g_batch).item() * (hi - lo)
        self._num_keep = max(self.min_keep, int(self.N * (1.0 - ratio)))

        ctx_list, tgt_list = [], []
        for b in range(B):
            g = torch.Generator()
            g.manual_seed(seed * B + b)
            ctx, tgt = self._sample_masks(imgs[b], g)
            ctx_list.append(ctx)
            tgt_list.append(tgt)

        min_ctx = min(c.numel() for c in ctx_list)
        min_tgt = min(t.numel() for t in tgt_list)
        ctx_tensor = torch.stack([c[:min_ctx] for c in ctx_list])   # (B, min_ctx)
        tgt_tensor = torch.stack([t[:min_tgt] for t in tgt_list])   # (B, min_tgt)

        return collated_batch, [ctx_tensor], [tgt_tensor]
