# Copyright (C) 2026. 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

import torch
from torch.utils.data import default_collate

from src.masks.base import BaseMaskCollator

logger = getLogger()


class GridMaskCollator(BaseMaskCollator):
    """Regular-grid (strided) masking.

    Divides the patch grid into non-overlapping ``stride × stride`` cells and
    retains one patch per cell as context (the top-left corner of each cell).
    All remaining patches form the prediction target.

    This gives a fixed context ratio of roughly ``1 / stride**2``:
      - stride=2 → ~25% context, ~75% target
      - stride=3 → ~11% context, ~89% target
      - stride=4 → ~6% context,  ~94% target

    Unlike random or cluster masking, the context mask is deterministic and
    identical for every image in the batch.

    Args:
        input_size: image side length in pixels, int or (H, W).
        patch_size: patch side length in pixels.
        stride: cell side length in patches.  Context = one patch per cell.
    """

    def __init__(
        self,
        input_size=224,
        patch_size: int = 16,
        stride: int = 2,
    ):
        super().__init__(input_size=input_size, patch_size=patch_size)
        self.stride = stride

        # Pre-compute fixed context and target indices (same for every image)
        row_idx = torch.arange(0, self.height, stride)
        col_idx = torch.arange(0, self.width, stride)
        rows, cols = torch.meshgrid(row_idx, col_idx, indexing='ij')
        ctx_flat = (rows * self.width + cols).flatten().sort()[0]   # (N_ctx,)

        in_ctx = torch.zeros(self.N, dtype=torch.bool)
        in_ctx[ctx_flat] = True
        tgt_flat = torch.arange(self.N)[~in_ctx]                    # (N_tgt,)

        self._ctx_indices = ctx_flat   # (N_ctx,) — fixed
        self._tgt_indices = tgt_flat   # (N_tgt,) — fixed

        logger.info(
            f'GridMaskCollator: stride={stride}, '
            f'N_ctx={ctx_flat.numel()}, N_tgt={tgt_flat.numel()}, '
            f'context_ratio={ctx_flat.numel() / self.N:.3f}'
        )

    def __call__(self, batch):
        """Collate a batch and produce grid-based encoder / predictor masks.

        The grid pattern is fixed (same for every image).

        Returns:
            collated_batch: standard default_collate output.
            collated_masks_enc: list with one LongTensor (B, N_ctx).
            collated_masks_pred: list with one LongTensor (B, N_tgt).
        """
        B = len(batch)
        collated_batch = default_collate(batch)
        self.step()   # keep counter consistent with other collators

        # Broadcast fixed masks to batch dimension (.contiguous() required: expand()
        # returns a non-contiguous view with shared memory, which pin_memory rejects)
        ctx_tensor = self._ctx_indices.unsqueeze(0).expand(B, -1).contiguous()   # (B, N_ctx)
        tgt_tensor = self._tgt_indices.unsqueeze(0).expand(B, -1).contiguous()   # (B, N_tgt)

        return collated_batch, [ctx_tensor], [tgt_tensor]
