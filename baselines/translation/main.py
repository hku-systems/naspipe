# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from collections import OrderedDict
import importlib
import json
import os
import shutil
import sys
import time
import itertools
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from fairseq.meters import AverageMeter
from fairseq import data, distributed_utils, options, progress_bar, tasks, utils, bleu, tokenizer
from fairseq.tasks import TASK_REGISTRY
from fairseq.criterions import CRITERION_REGISTRY
from fairseq.optim.lr_scheduler import LR_SCHEDULER_REGISTRY

from transformer_oneshot import TransformerModel

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default="data/wmt14_en_de_joined_dict", type=str,
                    help='path to dataset')
parser.add_argument('--max-tokens', default=10240, type=int,
                    metavar='N', help='maximum number of tokens in a batch (default: 2560)')
parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--lr', default=1.9e-3, type=float, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', default=0.99, type=float, metavar='M',
                    help='momentum factor')
parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                    help='weight decay')
parser.add_argument('--adam-betas', default="(0.9, 0.98)", metavar='B',
                    help='betas for Adam optimizer')
parser.add_argument('--adam-eps', default=1e-9, type=float, metavar='D',
                    help='epsilon for Adam optimizer')

parser.add_argument('--train-subset', default='train', metavar='SPLIT',
                    choices=['train', 'valid', 'test'],
                    help='data subset to use for training (train, valid, test)')
parser.add_argument('--valid-subset', default='valid', metavar='SPLIT',
                    help='comma separated list of data subsets to use for validation'
                    ' (train, valid, valid1, test, test1)')
parser.add_argument('--max-sentences-valid', type=int, metavar='N',
                    help='maximum number of sentences in a validation batch'
                    ' (defaults to --max-sentences)')

parser.add_argument('--task', metavar='TASK', default='translation', choices=TASK_REGISTRY.keys(),
                    help='task: {} (default: {})'.format(', '.join(TASK_REGISTRY.keys()), 'translation'))
parser.add_argument('--source-lang', default=None, metavar='SRC',
                    help='source language')
parser.add_argument('--target-lang', default=None, metavar='TARGET',
                    help='target language')
parser.add_argument('--raw-text', action='store_true',
                    help='load raw text dataset')
parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                    help='pad the source on the left (default: True)')
parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                    help='pad the target on the left (default: False)')
parser.add_argument('--pad-sequence', default=1, type=int, metavar='N',
                    help='Pad sequences to a multiple of N')

parser.add_argument('--criterion', default='label_smoothed_cross_entropy_original', metavar='CRIT',
                    choices=CRITERION_REGISTRY.keys(),
                    help='training criterion: {} (default: label_smoothed_cross_entropy)'.format(
                    ', '.join(CRITERION_REGISTRY.keys())),)
parser.add_argument('--label-smoothing', default=0.1, type=float, metavar='D',
                    help='epsilon for label smoothing, 0 means no label smoothing')

parser.add_argument('--warmup-updates', default=4000, type=int, metavar='N',
                    help='warmup the learning rate linearly for the first N updates')
parser.add_argument('--warmup-init-lr', default=-1, type=float, metavar='LR',
                    help='initial learning rate during warmup phase; default is 0')
parser.add_argument('--lr-scheduler', default='inverse_sqrt',
                    help='learning rate scheduler: {} (default: inverse_sqrt)'.format(
                         ', '.join(LR_SCHEDULER_REGISTRY.keys())))

def load_dataset_splits(task, splits):
    for split in splits:
        if split == 'train':
            task.load_dataset(split, combine=True)
        else:
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')
                try:
                    task.load_dataset(split_k, combine=False)
                except FileNotFoundError as e:
                    if k > 0:
                        break
                    raise e

def main():

    args = parser.parse_args()

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)
    model = TransformerModel.build_model(args, task).cuda()
    criterion = task.build_criterion(args).cuda()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr, betas=eval(args.adam_betas),
        eps=args.adam_eps, weight_decay=args.weight_decay
    )


    # Load dataset splits
    load_dataset_splits(task, ['train', 'valid'])

    epoch_itr = data.EpochBatchIterator(
        dataset=task.dataset(args.train_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences_valid,
        max_positions=(args.max_source_positions, args.max_target_positions),
        ignore_invalid_inputs=True,
        required_batch_size_multiple=8,
        seed=1,
        num_shards=1,
        shard_id=0,
    )

    losses = AverageMeter()

    encoder_layer_forward = [AverageMeter() for _ in range(len(model.encoder.layers[0].layer))]
    decoder_layer_forward = [AverageMeter() for _ in range(len(model.decoder.layers[0].layer))]
    encoder_layer_backward = [AverageMeter() for _ in range(len(model.encoder.layers[0].layer))]
    decoder_layer_backward = [AverageMeter() for _ in range(len(model.decoder.layers[0].layer))]

    def measure_hook(forward, backward):

        def hook(module, input, output):
            for i, layer in enumerate(module.layer):

                if len(input) == 2:
                    x, _ = input
                else:
                    x, = input
                x = x.detach().clone().requires_grad_()

                # warm-up
                for _ in range(5):
                    if isinstance(layer, nn.MultiheadAttention):
                        out, _ = layer(x, x, x)
                    else:
                        out = layer(x)
                    torch.autograd.backward(out, out)

                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                for _ in range(50):
                    starter.record()
                    if isinstance(layer, nn.MultiheadAttention):
                        out, _ = layer(x, x, x)
                    else:
                        out = layer(x)
                    ender.record()
                    torch.cuda.synchronize()
                    forward[i].update(starter.elapsed_time(ender))

                    starter.record()
                    torch.autograd.backward(out, out)
                    ender.record()
                    torch.cuda.synchronize()
                    backward[i].update(starter.elapsed_time(ender))

        return hook

    for layer in model.encoder.layers:
        layer.register_forward_hook(measure_hook(encoder_layer_forward, encoder_layer_backward))

    for layer in model.decoder.layers:
        layer.register_forward_hook(measure_hook(decoder_layer_forward, decoder_layer_backward))

    embed_forward = AverageMeter()
    embed_backward = AverageMeter()

    def embed_hook(module, input, output):
        tokens, _ = input

        # warm-up
        for _ in range(5):
            x = module.embed_scale * module.embed_tokens(tokens)
            x += module.embed_positions(tokens)
            torch.autograd.backward(x, x)

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(50):
            starter.record()
            x = module.embed_scale * module.embed_tokens(tokens)
            x += module.embed_positions(tokens)
            ender.record()
            torch.cuda.synchronize()
            embed_forward.update(starter.elapsed_time(ender))

            starter.record()
            torch.autograd.backward(x, x)
            ender.record()
            torch.cuda.synchronize()
            embed_backward.update(starter.elapsed_time(ender))

    model.encoder.register_forward_hook(embed_hook)

    linear_forward = AverageMeter()
    linear_backward = AverageMeter()

    def linear_hook(module, input, output):
        _, encode_out = input
        encode_out = encode_out.detach().clone().requires_grad_()

        # warm-up
        for _ in range(5):
            x = encode_out.transpose(0, 1)
            out = F.linear(x, module.embed_out)
            torch.autograd.backward(out, out)

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(50):
            starter.record()
            x = encode_out.transpose(0, 1)
            out = F.linear(x, module.embed_out)
            ender.record()
            torch.cuda.synchronize()
            linear_forward.update(starter.elapsed_time(ender))

            starter.record()
            torch.autograd.backward(out, out)
            ender.record()
            torch.cuda.synchronize()
            linear_backward.update(starter.elapsed_time(ender))

    model.decoder.register_forward_hook(linear_hook)

    itr = epoch_itr.next_epoch_itr()
    max_positions = (args.max_source_positions, args.max_target_positions)
    for i, sample in enumerate(itr):
        sample = task.dataset('train').get_dummy_batch(args.max_tokens, max_positions)
        sample = utils.move_to_cuda(sample)
        loss, _, logging_output = criterion(model, sample)
        num_tokens = logging_output['ntokens']
        losses.update(loss.item() / num_tokens / math.log(2), num_tokens)
        if i % 100 == 0:
            print('Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(
                    loss=losses))
            print('Time: {forward_time.avg:.3f} ({backward_time.avg:.3f})'
                    '{forward_time_decoder.avg:.3f} ({backward_time_decoder.avg:.3f})'.format(
                    forward_time=encoder_layer_forward[0],
                    backward_time=encoder_layer_backward[0],
                    forward_time_decoder=decoder_layer_forward[-1],
                    backward_time_decoder=decoder_layer_backward[-1]))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        break

    stat = {i: {} for i in range(len(decoder_layer_forward))}
    for i, (f, b) in enumerate(zip(encoder_layer_forward, encoder_layer_backward)):
        stat[i]['encoder'] = {}
        stat[i]['encoder']['forward'] = f.avg
        stat[i]['encoder']['backward'] = b.avg

    for i, (f, b) in enumerate(zip(decoder_layer_forward, decoder_layer_backward)):
        stat[i]['decoder'] = {}
        stat[i]['decoder']['forward'] = f.avg
        stat[i]['decoder']['backward'] = b.avg

    stat['embed'] = {}
    stat['embed']['forward'] = embed_forward.avg
    stat['embed']['backward'] = embed_backward.avg

    stat['linear'] = {}
    stat['linear']['forward'] = linear_forward.avg
    stat['linear']['backward'] = linear_backward.avg

    with open('time.json', 'w') as file:
        json.dump(stat, file, indent=4)

if __name__ == '__main__':
    main()
