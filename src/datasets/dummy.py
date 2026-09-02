# Copyright (C) 2026. 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

import torch

logger = getLogger()


class DummyImageNet(torch.utils.data.Dataset):
    """Generates random (image, label) pairs that mimic ImageNet tensors.

    Used for local smoke-testing without access to real data.
    The transform pipeline is bypassed: images are returned as float tensors
    already normalised to [-1, 1] so they are compatible with the mask
    collator and model forward passes.
    """

    def __init__(self, crop_size=224, length=1024):
        self.crop_size = crop_size
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = torch.rand(3, self.crop_size, self.crop_size) * 2. - 1.
        return img, 0


def make_dummy_data(
    crop_size,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=0,
    world_size=1,
    rank=0,
    length=1024,
    drop_last=True,
):
    # Create dummy dataset and dataloader.
    dataset = DummyImageNet(crop_size=crop_size, length=length)
    logger.info(f'dummy dataset created ({length} synthetic images, crop {crop_size})')

    # Create distributed sampler
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    logger.info('dummy distributed sampler created, num_replicas: {}, rank: {}'.format(world_size, rank))
    
    # Create dataloader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('dummy data loader created, num_batches: {}'.format(len(data_loader)))
    return dataset, data_loader, dist_sampler
