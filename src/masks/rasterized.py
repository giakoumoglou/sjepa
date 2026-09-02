# Copyright (C) 2026. 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Rasterized (quadrant) masking.
https://arxiv.org/abs/2301.08243

The image is divided into four equal quadrants:

    ┌──────────┬──────────┐
    │          │          │
    │  Q0 (TL) │  Q1 (TR) │
    │          │          │
    ├──────────┼──────────┤
    │          │          │
    │  Q2 (BL) │  Q3 (BR) │
    │          │          │
    └──────────┴──────────┘

One quadrant is randomly chosen each batch as the context (visible to the
encoder).  The remaining three quadrants are three independent prediction
targets.  There is no overlap between context and any target block.

    Context ratio : 1/4 = 25 %
    Target ratio  : 3/4 = 75 % (split across 3 separate pred blocks)

The quadrant used as context is the same for every image in a batch
(determined by the batch seed), consistent with the per-batch block-size
convention used by MultiblockMaskCollator.

Usage (called from train.py):

    from src.masks.rasterized import RasterizedMaskCollator
    collator = RasterizedMaskCollator(input_size=224, patch_size=16)
    # collator(batch) -> (collated_batch, [ctx_tensor], [tgt_q1, tgt_q2, tgt_q3])
"""

from __future__ import annotations

from logging import getLogger

import torch
from torch.utils.data import default_collate

from src.masks.base import BaseMaskCollator

logger = getLogger()


class RasterizedMaskCollator(BaseMaskCollator):
    """Quadrant-based masking for reconstruction pre-training.

    Divides the (height × width) patch grid into four equal quadrants and
    uses one randomly-selected quadrant as the encoder context.  The other
    three quadrants become three separate prediction targets.

    Args:
        input_size: image side length in pixels, int or (H, W) tuple.
        patch_size: patch side length in pixels.
    """

    def __init__(
        self,
        input_size=224,
        patch_size: int = 16,
    ) -> None:
        super().__init__(input_size=input_size, patch_size=patch_size)

        # Midpoints — floor division so odd grids put the extra row/col in the
        # lower-right quadrants, keeping all four sets non-overlapping.
        h_mid = self.height // 2
        w_mid = self.width  // 2

        def _quadrant(row_start, row_end, col_start, col_end) -> torch.Tensor:
            rows = torch.arange(row_start, row_end)
            cols = torch.arange(col_start, col_end)
            r, c = torch.meshgrid(rows, cols, indexing='ij')
            return (r * self.width + c).flatten().sort()[0]

        self._quadrants: list[torch.Tensor] = [
            _quadrant(0,      h_mid,        0,      w_mid),        # Q0: top-left
            _quadrant(0,      h_mid,        w_mid,  self.width),   # Q1: top-right
            _quadrant(h_mid,  self.height,  0,      w_mid),        # Q2: bottom-left
            _quadrant(h_mid,  self.height,  w_mid,  self.width),   # Q3: bottom-right
        ]

        sizes = [q.numel() for q in self._quadrants]
        logger.info(
            f'rasterized mask collator: '
            f'grid={self.height}x{self.width} '
            f'quadrant_sizes={sizes} '
            f'context_ratio={sizes[0] / self.N:.3f}'
        )

    def __call__(self, batch):
        """Collate a batch and produce quadrant-based encoder / predictor masks.

        The context quadrant is drawn once per batch (same index for all
        images).  The three target quadrants are returned as three separate
        tensors so the training loop can compute one prediction per quadrant.

        Returns:
            collated_batch:    standard default_collate output.
            collated_masks_enc: list with one LongTensor (B, N_ctx) —
                                the context quadrant patches.
            collated_masks_pred: list of three LongTensors (B, N_tgt) —
                                 one per prediction quadrant.
        """
        B = len(batch)
        collated_batch = default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        ctx_q = int(torch.randint(4, (1,), generator=g).item())

        ctx_indices = self._quadrants[ctx_q]                              # (N_ctx,)
        tgt_indices = [self._quadrants[q] for q in range(4) if q != ctx_q]  # 3 × (N_tgt,)

        ctx_tensor  = ctx_indices.unsqueeze(0).expand(B, -1).contiguous()
        tgt_tensors = [t.unsqueeze(0).expand(B, -1).contiguous() for t in tgt_indices]

        return collated_batch, [ctx_tensor], tgt_tensors
