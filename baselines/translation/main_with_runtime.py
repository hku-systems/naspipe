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
import subprocess

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

sys.path.append("..")
import runtime

from fairseq import data, distributed_utils, options, progress_bar, tasks, utils, bleu, tokenizer
from fairseq.fp16_trainer import FP16Trainer
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter, TimeMeter
from fairseq.sequence_generator import SequenceGenerator
from fairseq.data import dictionary
from fairseq.tasks import TASK_REGISTRY
from fairseq.criterions import CRITERION_REGISTRY
from fairseq.optim import lr_scheduler
from fairseq.optim import adam, sgd
from fairseq.optim.lr_scheduler import LR_SCHEDULER_REGISTRY

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_dir', type=str,
                    help='path to dataset')
parser.add_argument('--distributed_backend', type=str,
                    help='distributed backend to use (gloo|nccl)')
parser.add_argument('--module', '-m', required=True,
                    help='name of module that contains model and tensor_shapes definition')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=8, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--max-tokens', default=2560, type=int,
                    metavar='N', help='maximum number of tokens in a batch (default: 2560)')

parser.add_argument('--batch_size', default=192, type=int,
                    metavar='N', help='batch size')

parser.add_argument('--rep', default=2, type=int,
                    metavar='N', help='tokens')
parser.add_argument('--input_path', default="config_4_3.json", type=str,
                    help="Path of configuration file")

parser.add_argument('--sys', default="pipedream", type=str,
                    help="sys type")

parser.add_argument('--grad-clip', default=5.0, type=float,
                    help='enabled gradient clipping and sets maximum gradient norm value')
parser.add_argument('--eval-batch-size', default=100, type=int,
                    help='eval mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_policy', default='step', type=str,
                    help='policy for controlling learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--fp16', action='store_true',
                    help='train model in fp16 precision')
parser.add_argument('--loss_scale', type=float, default=1,
                    help='static loss scale, positive power of 2 to improve fp16 convergence')
parser.add_argument('--master_addr', default=None, type=str,
                    help="IP address of master (machine with rank 0)")
parser.add_argument('--config_path', default=None, type=str,
                    help="Path of configuration file")
parser.add_argument('--no_input_pipelining', action='store_true',
                    help="No pipelining of inputs")
parser.add_argument('--rank', default=None, type=int,
                    help="Rank of worker")
parser.add_argument('--local_rank', default=0, type=int,
                    help="Local rank of worker")
parser.add_argument('--forward_only', action='store_true',
                    help="Run forward pass only")
parser.add_argument('--num_minibatches', default=1000, type=int,
                    help="Number of minibatches to run")
parser.add_argument('--resume', action='store_true',
                    help='Resume from latest checkpoint')
parser.add_argument('--checkpoint_dir', default='', type=str, metavar='PATH',
                    help='path to directory to save checkpoints')
parser.add_argument('--checkpoint_dir_not_nfs', action='store_true',
                    help='checkpoint dir is not on a shared NFS server')
parser.add_argument('-s', '--synthetic_data', action='store_true',
                    help="Use synthetic data")
parser.add_argument('-v', '--verbose_frequency', default=0, type=int, metavar='N',
                    help="Log verbose information")
parser.add_argument('--num_ranks_in_server', default=1, type=int,
                    help="number of gpus per machine")

# Recompute tensors from forward pass, instead of saving them.
parser.add_argument('--recompute', action='store_true',
                    help='Recompute tensors in backward pass')
# Macrobatching reduces the number of weight versions to save,
# by not applying updates every minibatch.
parser.add_argument('--macrobatch', action='store_true',
                    help='Macrobatch updates to save memory')

parser.add_argument('--skip-invalid-size-inputs-valid-test', action='store_true',
                    help='ignore too long or too short lines in valid and test set')
parser.add_argument('--max-sentences', '--batch-size', type=int, metavar='N',
                    help='maximum number of sentences in a batch')
parser.add_argument('--sentencepiece', action='store_true',
                    help='use when dataset uses sentencepiece encoding')

parser.add_argument('--train-subset', default='train', metavar='SPLIT',
                    choices=['train', 'valid', 'test'],
                    help='data subset to use for training (train, valid, test)')
parser.add_argument('--valid-subset', default='valid', metavar='SPLIT',
                    help='comma separated list of data subsets to use for validation'
                    ' (train, valid, valid1, test, test1)')
parser.add_argument('--max-sentences-valid', type=int, metavar='N',
                    help='maximum number of sentences in a validation batch'
                    ' (defaults to --max-sentences)')

parser.add_argument('--gen-subset', default='test', metavar='SPLIT',
                    help='data subset to generate (train, valid, test)')
parser.add_argument('--num-shards', default=1, type=int, metavar='N',
                    help='shard generation over N shards')
parser.add_argument('--shard-id', default=0, type=int, metavar='ID',
                    help='id of the shard to generate (id < num_shards)')

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

parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                    help='max number of tokens in the source sequence')
parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                    help='max number of tokens in the target sequence')
                    
parser.add_argument('--pad-sequence', default=1, type=int, metavar='N',
                    help='Pad sequences to a multiple of N')

parser.add_argument('--criterion', default='label_smoothed_cross_entropy', metavar='CRIT',
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


best_prec1 = 0


# Helper methods.
def is_first_stage():
    return args.stage is None or (args.stage == 0)

def is_last_stage():
    return args.stage is None or (args.stage == (args.num_stages-1))

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
    global args, best_prec1
    args = parser.parse_args()
    args.data = args.data_dir

    os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.local_rank}"

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    criterion = task.build_criterion(args)

    # create stages of the model
    module = importlib.import_module(args.module)
    args.arch = module.arch()
    model = module.model(criterion)

    max_positions = (args.max_source_positions, args.max_target_positions)

    training_tensor_shapes = {'input0': [60, 128], 'input1': [60, 128], 'target': [60, 128], 'ntokens': [1], 'control': [1, 100], 'out0': [128, 60, 1024], 'out1': [128, 60, 1024], 'out2': [128, 60, 1024], 'out3': [128, 60, 1024], 'out4': [128, 60, 1024], 'out5': [128, 60, 1024], 'out6': [128, 60, 1024], 'out7': [128, 60, 1024],'out8': [128, 60, 1024],'out9': [128, 60, 1024],'out10': [128, 60, 1024],'out11': [128, 60, 1024],'out12': [128, 60, 1024],'out13': [128, 60, 1024],'out14': [128, 60, 1024],'out15': [128, 60, 1024], 'out16': [60, 128, 33712]}
    dtypes = {'input0': torch.int64, 'input1': torch.int64, 'target': torch.int64, 'ntokens': torch.float32, 'control': torch.int32, 'out0': torch.float32, 'out1': torch.float32, 'out2': torch.float32, 'out3': torch.float32, 'out4': torch.float32, 'out5': torch.float32, 'out6': torch.float32,'out7': torch.float32,'out8': torch.float32,'out9': torch.float32,'out10': torch.float32,'out11': torch.float32,'out12': torch.float32,'out13': torch.float32,'out14': torch.float32,'out15': torch.float32,'out16': torch.float32,'out17': torch.float32,'out18': torch.float32,'out19': torch.float32}
    inputs_module_destinations = {'input0': 0, 'input1': 0}
    target_tensor_names = {'target', 'ntokens'}

    eval_tensor_shapes = {}
    for key in training_tensor_shapes:
        eval_tensor_shapes[key] = tuple(
            training_tensor_shapes[key])
        training_tensor_shapes[key] = tuple(
            training_tensor_shapes[key])

    configuration_maps = {
        'module_to_stage_map': None,
        'stage_to_rank_map': None,
        'stage_to_depth_map': None
    }
    if args.config_path is not None:
        json_config_file = json.load(open(args.config_path, 'r'))
        configuration_maps['module_to_stage_map'] = json_config_file.get("module_to_stage_map", None)
        configuration_maps['stage_to_rank_map'] = json_config_file.get("stage_to_rank_map", None)
        configuration_maps['stage_to_rank_map'] = {
            int(k): v for (k, v) in configuration_maps['stage_to_rank_map'].items()}
        configuration_maps['stage_to_depth_map'] = json_config_file.get("stage_to_depth_map", None)

    r = runtime.StageRuntime(
        model=model, distributed_backend=args.distributed_backend,
        fp16=args.fp16, loss_scale=args.loss_scale,
        training_tensor_shapes=training_tensor_shapes,
        eval_tensor_shapes=eval_tensor_shapes,
        training_tensor_dtypes=dtypes,
        inputs_module_destinations=inputs_module_destinations,
        target_tensor_names=target_tensor_names,
        configuration_maps=configuration_maps,
        master_addr=args.master_addr,
        rank=args.rank, local_rank=args.local_rank,
        num_ranks_in_server=args.num_ranks_in_server,
        verbose_freq=args.verbose_frequency,
        model_type=runtime.TRANSLATION,
        enable_recompute=args.recompute, 
        batch_s=args.batch_size,
        rep=args.rep, input_path=args.input_path)

    # stage needed to determine if current stage is the first stage
    # num_stages needed to determine if current stage is the last stage
    # num_ranks needed to determine number of warmup_minibatches in case of pipelining
    args.stage = r.stage
    args.num_stages = r.num_stages
    args.num_ranks = r.num_ranks
    if not is_first_stage():
        args.synthetic_data = True

    # define optimizer
    if args.no_input_pipelining:
        num_versions = 1
    else:
        # number of versions is the total number of machines following the current
        # stage, shared amongst all replicas in this stage
        num_versions = r.num_warmup_minibatches + 1

    # if specified, resume from checkpoint
    if args.resume:
        checkpoint_file_path = os.path.join(args.checkpoint_dir,
                                            f"checkpoint.{r.stage}.pth.tar.epoch.{args.start_epoch}")
        assert os.path.isfile(checkpoint_file_path)
        print("=> loading checkpoint '{}'".format(checkpoint_file_path))
        checkpoint = torch.load(checkpoint_file_path)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        r.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(checkpoint_file_path, checkpoint['epoch']))

    # TODO: make this configurable by args
    use_adam_optimizer = True
    if use_adam_optimizer:
        optimizer = adam.Adam(
            r.master_parameters,
            lr=args.lr, betas=(0.9, 0.98),
            weight_decay=args.weight_decay
        )
    else:
        optimizer = sgd.SGD(
            r.master_parameters,
            lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    scheduler = lr_scheduler.build_lr_scheduler(args, optimizer)

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True

    epoch_itr = None

    distributed_sampler = False
    if configuration_maps['stage_to_rank_map'] is not None:
        num_ranks_in_first_stage = len(configuration_maps['stage_to_rank_map'][0])
        if num_ranks_in_first_stage > 1:
            distributed_sampler = True

    for epoch in range(args.start_epoch, args.epochs):
        if distributed_sampler:
            train_loader.sampler.set_epoch(epoch)

        # train or run forward pass only for one epoch
        if args.forward_only:
            validate(val_loader, r, epoch)
        else:
            if args.sys == "gpipe":
                train_gpipe(epoch_itr, r, optimizer, epoch, scheduler)
            elif args.sys == "vpipe":
                train_gpipe(epoch_itr, r, optimizer, epoch, scheduler)
            elif args.sys == "pipedream":
                train_pipedream(epoch_itr, r, optimizer, epoch, scheduler)
        break


s = [[], []]



def train_pipedream(train_loader, r, optimizer, epoch, lr_scheduler):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    n = args.num_minibatches# r.num_iterations(loader_size=len(train_loader))
    if args.num_minibatches is not None:
        n = min(n, args.num_minibatches)

    # accumulation = 32
    # n -= (n % accumulation)
    # assert n % accumulation == 0

    r.train(n)
    train_loader="performance"
    if not is_first_stage(): train_loader = None
    r.set_loader(train_loader)#r.set_loader(train_loader)

    end = time.time()
    epoch_start_time = time.time()

    if args.no_input_pipelining:
        num_warmup_minibatches = 0
    else:
        num_warmup_minibatches = r.num_warmup_minibatches

    if args.verbose_frequency > 0:
        print("Letting in %d warm-up minibatches" % num_warmup_minibatches)
        print("Running training for %d minibatches" % n)

    #r.set_loss_scale(4 / accumulation)
    total_updates = n #// accumulation
    for i in range(num_warmup_minibatches):
        r.receive_tensors_forward()
        r.run_forward()
    for t in range(1, total_updates+1):
        
        # start num_warmup_minibatches forward passes
        r.receive_tensors_forward()

        torch.cuda.synchronize()
        fstart = time.time()
        r.run_forward()
        torch.cuda.synchronize()
        fend = time.time()

                #print("Stage ", r.stage, ": forward ", seq_id, " buffer size ", len(r.buffer_list), flush=True)

        if is_last_stage():
        # measure accuracy and record loss
            output, target, loss, num_tokens = r.output, r.target, r.loss.item(), r.num_tokens()
            # print(loss, num_tokens)
            losses.update(loss / num_tokens / math.log(2), num_tokens)


        r.receive_tensors_backward() 

        torch.cuda.synchronize()
        bstart = time.time()
        r.run_backward()
        torch.cuda.synchronize()
        bend = time.time()

        bwd_seq_id = r.current_bwd_idx
                #print("Stage ", r.stage, ": backward finish ", bwd_seq_id, flush=True)

        if is_last_stage():
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            epoch_time = (end - epoch_start_time) / 3600.0
            full_epoch_time = (epoch_time / float(t)) * float(n)

            if t % args.print_freq == 0:
                print('Stage: [{0}] Epoch: [{1}][{2}/{3}]\t'
                    'Time({timestamp}): {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Epoch time [hr]: {epoch_time:.3f} ({full_epoch_time:.3f})\t'.format(
                    args.stage, epoch, t, n, timestamp=time.time(), batch_time=batch_time,
                    epoch_time=epoch_time, full_epoch_time=full_epoch_time))
                    #loss=losses, # top1=top1, top5=top5,
                    #memory=(float(torch.cuda.memory_allocated()) / 10**9),
                    #cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
                import sys; sys.stdout.flush()
        else:
            if t % args.print_freq == 0:
                print('Stage: [{0}] Epoch: [{1}][{2}/{3}]'.format(
                    args.stage, epoch, t, n, memory=(float(torch.cuda.memory_allocated()) / 10**9),
                    cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
                import sys; sys.stdout.flush()

                
            #else:
            #    print("Stage ", r.stage, ": No backward, continue forward")


            # if i == 500 and args.local_rank == 0:
            #     subprocess.Popen(['python', 'usage.py', 'gpu.log'])

        #finish remaining backward passes


        # r.modules_with_dependencies.modules()[0].grad_cpu()
        #r.modules_with_dependencies.modules()[0].param_cuda()
        # optimizer.step()
        # print("Stage ", r.stage, ": step")
        # r.modules_with_dependencies.modules()[0].param_cpu()
        if args.fp16:
            r.zero_grad()
        else:
            # optimizer.zero_grad()
            for group in optimizer.param_groups:
                for p in group['params']:   
                    if p.grad is not None:
                        p.grad = None
                        #p.data = p.data.cpu()
        num_updates = epoch * total_updates + t + 1
        lr_scheduler.step_update(num_updates)

    for i in range(num_warmup_minibatches):
        r.receive_tensors_backward()
        r.run_backward()
    # wait for all helper threads to complete
    r.wait()

    print("Epoch %d: %.3f seconds" % (epoch, time.time() - epoch_start_time))
    print("Epoch start time: %.3f, epoch end time: %.3f" % (epoch_start_time, time.time()))

def train_gpipe(train_loader, r, optimizer, epoch, lr_scheduler):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    num_stages = r.num_stages
    # switch to train mode
    n = args.num_minibatches# r.num_iterations(loader_size=len(train_loader))
    if args.num_minibatches is not None:
        n = min(n, args.num_minibatches)

    r.train(n)
    train_loader="performance"
    if not is_first_stage(): train_loader = None
    r.set_loader(train_loader)#r.set_loader(train_loader)

    end = time.time()
    epoch_start_time = time.time()


    num_warmup_minibatches = r.num_warmup_minibatches

    if args.verbose_frequency > 0:
        print("Letting in %d warm-up minibatches" % num_warmup_minibatches)
        print("Running training for %d minibatches" % n)

    #r.set_loss_scale(4 / accumulation)
    total_updates = n #// accumulation

    t = 1
    while t < total_updates+1:
        
        # start num_warmup_minibatches forward passes

        for i in range(num_warmup_minibatches):
            r.receive_tensors_forward()
            r.run_forward()
            t += 1

        for j in range(num_stages - num_warmup_minibatches):
            r.receive_tensors_forward()

            torch.cuda.synchronize()
            fstart = time.time()
            r.run_forward()
            t += 1
            torch.cuda.synchronize()
            fend = time.time()
                #print("Stage ", r.stage, ": forward ", seq_id, " buffer size ", len(r.buffer_list), flush=True)
            if is_last_stage():
        # measure accuracy and record loss
                output, target, loss, num_tokens = r.output, r.target, r.loss.item(), r.num_tokens()
            # print(loss, num_tokens)
                losses.update(loss / num_tokens / math.log(2), num_tokens)


            r.receive_tensors_backward() 

            torch.cuda.synchronize()
            bstart = time.time()
            r.run_backward()
            torch.cuda.synchronize()
            bend = time.time()

            bwd_seq_id = r.current_bwd_idx
                #print("Stage ", r.stage, ": backward finish ", bwd_seq_id, flush=True)

            if is_last_stage():
            # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                epoch_time = (end - epoch_start_time) / 3600.0
                full_epoch_time = (epoch_time / float(t)) * float(n)

                if t % args.print_freq == 0:
                    print('Stage: [{0}] Epoch: [{1}][{2}/{3}]\t'
                        'Time({timestamp}): {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Epoch time [hr]: {epoch_time:.3f} ({full_epoch_time:.3f})\t'.format(
                        args.stage, epoch, t, n, timestamp=time.time(), batch_time=batch_time,
                        epoch_time=epoch_time, full_epoch_time=full_epoch_time))
                    #loss=losses, # top1=top1, top5=top5,
                    #memory=(float(torch.cuda.memory_allocated()) / 10**9),
                    #cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
                    import sys; sys.stdout.flush()
            else:
                if t % args.print_freq == 0:
                    print('Stage: [{0}] Epoch: [{1}][{2}/{3}]'.format(
                        args.stage, epoch, t, n, memory=(float(torch.cuda.memory_allocated()) / 10**9),
                        cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
                    import sys; sys.stdout.flush()

                
            #else:
            #    print("Stage ", r.stage, ": No backward, continue forward")


            # if i == 500 and args.local_rank == 0:
            #     subprocess.Popen(['python', 'usage.py', 'gpu.log'])


            if args.fp16:
                r.zero_grad()
            else:
            # optimizer.zero_grad()
                for group in optimizer.param_groups:
                    for p in group['params']:   
                        if p.grad is not None:
                            p.grad = None
                        if args.sys == "vpipe":
                            if hasattr(p, 'choice_'):
                                p.data = p.data.cpu()
            num_updates = epoch * total_updates + t + 1
            lr_scheduler.step_update(num_updates)

        for i in range(num_warmup_minibatches):
            r.receive_tensors_backward()
            r.run_backward()
            for group in optimizer.param_groups:
                for p in group['params']:   
                    if p.grad is not None:
                        p.grad = None
                    if args.sys == "vpipe":
                        if hasattr(p, 'choice_'):
                            p.data = p.data.cpu()
            # wait for all helper threads to complete

        #t = t + num_stages


    r.wait()

    print("Epoch %d: %.3f seconds" % (epoch, time.time() - epoch_start_time))
    print("Epoch start time: %.3f, epoch end time: %.3f" % (epoch_start_time, time.time()))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
