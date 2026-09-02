# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import multiprocessing as mp

import pprint
import yaml

from src.utils.distributed import init_distributed, find_free_port
from src.train import main as app_main

parser = argparse.ArgumentParser()
parser.add_argument('--fname', type=str, default='configs.yaml', help='name of config file to load')
parser.add_argument('--devices', type=str, nargs='+', default=['cuda:0'], help='which devices to use on local machine')
parser.add_argument('--stochastic', action='store_true', default=False, help='override: enable stochastic predictor, no free bits')
parser.add_argument('--free_bits', type=float, default=None, metavar='LAMBDA', help='override: enable stochastic predictor with free bits (nats/dim, e.g. 0.5)')


def process_main(rank, fname, world_size, devices, port, overrides):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])
    os.environ['MASTER_ADDR'] = 'localhost'

    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called_params {fname}')

    # -- load script params
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')

    # -- apply CLI overrides (after YAML so flags win)
    if overrides.get('stochastic'):
        params.setdefault('meta', {})['stochastic'] = True
        params['meta'].setdefault('kl_free_bits', 0.0)
        logger.info('override: stochastic=True, kl_free_bits=0.0')
    if overrides.get('free_bits') is not None:
        params.setdefault('meta', {})['stochastic'] = True
        params['meta']['kl_free_bits'] = overrides['free_bits']
        logger.info(f'override: stochastic=True, kl_free_bits={overrides["free_bits"]}')

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(params)

    world_size, rank = init_distributed(
        port=port,
        rank_and_world_size=(rank, world_size),
    )
    logger.info(f'running... (rank: {rank}/{world_size})')
    app_main(args=params)


if __name__ == '__main__':
    args = parser.parse_args()

    overrides = {}
    if args.stochastic:
        overrides['stochastic'] = True
    if args.free_bits is not None:
        overrides['free_bits'] = args.free_bits

    num_gpus = len(args.devices)
    port = find_free_port()

    mp.set_start_method('spawn')

    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices, port, overrides)
        ).start()
