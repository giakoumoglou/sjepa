# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from logging import getLogger

import torch
from torch.utils.data import default_collate

from src.masks.base import BaseMaskCollator

logger = getLogger()


class MultiblockMaskCollator(BaseMaskCollator):
    """I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture
    https://arxiv.org/abs/2301.08243

    Samples ``npred`` rectangular target blocks and ``nenc`` rectangular
    context blocks per image.  Context blocks are constrained to not overlap
    with target blocks when ``allow_overlap=False``.

    Args:
        input_size: image side length in pixels, int or (H, W).
        patch_size: patch side length in pixels.
        enc_mask_scale: (min, max) fraction of patches for each context block.
        pred_mask_scale: (min, max) fraction of patches for each target block.
        aspect_ratio: (min, max) aspect-ratio for sampled blocks.
        nenc: number of context (encoder) blocks.
        npred: number of target (predictor) blocks.
        min_keep: minimum patches required for a valid block.
        allow_overlap: if True, context blocks may overlap with target blocks.
    """

    def __init__(
        self,
        input_size=224,
        patch_size: int = 16,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        nenc: int = 1,
        npred: int = 2,
        min_keep: int = 4,
        allow_overlap: bool = False,
    ):
        super().__init__(input_size=input_size, patch_size=patch_size)
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1
        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            N = max(int(len(acceptable_regions) - tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]

        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top + h, left:left + w] = 1
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(
                        f'Mask generator says: "Valid mask not found, '
                        f'decreasing acceptable-regions [{tries}]"'
                    )
        mask = mask.squeeze()
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top + h, left:left + w] = 0
        return mask, mask_complement

    def __call__(self, batch):
        """Collate a batch and produce multi-block encoder / predictor masks.

        Returns:
            collated_batch: standard default_collate output.
            collated_masks_enc: list of ``nenc`` LongTensors, each (B, N_ctx).
            collated_masks_pred: list of ``npred`` LongTensors, each (B, N_tgt).
        """
        B = len(batch)
        collated_batch = default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(B):
            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions = None
            except Exception as e:
                logger.warning(f'encountered exception in mask-generator {e}')

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = default_collate(collated_masks_pred)
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred


# Backward-compatible alias used by existing train.py imports
MaskCollator = MultiblockMaskCollator
