# Copyright (C) 2026. 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from multiprocessing import Value
from logging import getLogger

logger = getLogger()


class BaseMaskCollator(object):
    """Shared infrastructure for all mask collators.

    Subclasses inherit:
      - self.height, self.width, self.N  — patch grid dimensions
      - self._itr_counter               — process-safe iteration counter
      - self.step()                     — increment and return counter

    All collators share the same __call__ contract:
        (collated_batch, collated_masks_enc, collated_masks_pred)
    where each mask list contains LongTensors of flat patch indices.
    """

    def __init__(
        self,
        input_size=224,
        patch_size=16,
    ):
        super().__init__()
        if not isinstance(input_size, tuple):
            input_size = (int(input_size), int(input_size))
        self.patch_size = patch_size
        self.height = input_size[0] // patch_size
        self.width = input_size[1] // patch_size
        self.N = self.height * self.width
        self._itr_counter = Value('i', -1)

    def step(self) -> int:
        """Increment and return the iteration counter (process-safe)."""
        ctr = self._itr_counter
        with ctr.get_lock():
            ctr.value += 1
            return int(ctr.value)

    def __call__(self, batch):
        raise NotImplementedError
