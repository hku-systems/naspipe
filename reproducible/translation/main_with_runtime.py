# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse


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
parser.add_argument('--grad-clip', default=5.0, type=float,
                    help='enabled gradient clipping and sets maximum gradient norm value')
parser.add_argument('--eval-batch-size', default=100, type=int,
                    help='eval mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_policy', default='step', type=str,
                    help='policy for controlling learning rate')
parser.add_argument('--momentum', default=0.0, type=float, metavar='M',
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
parser.add_argument('--num_minibatches', default=None, type=int,
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

parser.add_argument('--task', metavar='TASK', default='translation')

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
                    help='training criterion (default: label_smoothed_cross_entropy)')
parser.add_argument('--label-smoothing', default=0.1, type=float, metavar='D',
                    help='epsilon for label smoothing, 0 means no label smoothing')

parser.add_argument('--warmup-updates', default=4000, type=int, metavar='N',
                    help='warmup the learning rate linearly for the first N updates')
parser.add_argument('--warmup-init-lr', default=-1, type=float, metavar='LR',
                    help='initial learning rate during warmup phase; default is 0')
parser.add_argument('--lr-scheduler', default='inverse_sqrt',
                    help='learning rate scheduler (default: inverse_sqrt)')


args = parser.parse_args()



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

os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.local_rank}"

import torch
torch.use_deterministic_algorithms(True)

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

from fairseq import data, options, tasks, utils, tokenizer
#from fairseq.fp16_trainer import FP16Trainer
#from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter
from fairseq.data import dictionary
from fairseq.optim import lr_scheduler
from fairseq.optim import adam, sgd
from criterion import LabelSmoothedCrossEntropyCriterion


from fairseq.tasks import TASK_REGISTRY
from fairseq.criterions import CRITERION_REGISTRY
from fairseq.optim.lr_scheduler import LR_SCHEDULER_REGISTRY

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
    args.data = args.data_dir

    os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.local_rank}"

    # Enforce determinism 

    SEED = 123 # or whatever you choose
    # random.seed(SEED) # if you're using random
    # np.random.seed(123) # if you're using numpy
    torch.manual_seed(SEED) # torch.cuda.manual_seed_all(SEED) is not required
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load dataset splits
    load_dataset_splits(task, ['train', 'valid'])

    # Build criterion
    criterion = LabelSmoothedCrossEntropyCriterion(args, task)#task.build_criterion(args)

    # create stages of the model
    module = importlib.import_module(args.module)
    args.arch = module.arch()
    model = module.model(criterion)

    max_positions = (args.max_source_positions, args.max_target_positions)

    training_tensor_shapes = {'input0': [60, 128], 'input1': [60, 128], 'target': [60, 128], 'ntokens': [1], 'control': [1, 100], 'out0': [128, 60, 1024], 'out1': [128, 60, 1024], 'out2': [128, 60, 1024], 'out3': [128, 60, 1024], 'out4': [128, 60, 1024], 'out5': [128, 60, 1024], 'out6': [60, 128, 33712]}
    dtypes = {'input0': torch.int64, 'input1': torch.int64, 'target': torch.int64, 'ntokens': torch.float32, 'control': torch.int32, 'out0': torch.float32, 'out1': torch.float32, 'out2': torch.float32, 'out3': torch.float32, 'out4': torch.float32, 'out5': torch.float32, 'out6': torch.float32}
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
        enable_recompute=args.recompute)

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
    use_adam_optimizer = False
    if use_adam_optimizer:
        optimizer = adam.Adam(
            r.master_parameters,
            lr=args.lr, betas=(0.9, 0.98),
            weight_decay=args.weight_decay
        )
    else:
        args.lr = [0.001]
        optimizer = sgd.SGD(
            args,
            r.master_parameters
        )

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True

    epoch_itr = task.get_batch_iterator(
        dataset=task.dataset(args.train_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences_valid,
        max_positions=max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=32,
        seed=1,
        num_shards=1,
        shard_id=0
    )

    # epoch_itr = data.CountingIterator(
    #     task.dataset(args.train_subset)
    # )
    distributed_sampler = False
    if configuration_maps['stage_to_rank_map'] is not None:
        num_ranks_in_first_stage = len(configuration_maps['stage_to_rank_map'][0])
        if num_ranks_in_first_stage > 1:
            distributed_sampler = True

    r.set_share()

    for epoch in range(args.start_epoch, args.epochs):
        if distributed_sampler:
            train_loader.sampler.set_epoch(epoch)

        # train or run forward pass only for one epoch
        if args.forward_only:
            validate(val_loader, r, epoch)
        else:
            train(epoch_itr, r, optimizer, epoch)

            # evaluate on validation set
            # prec1 = validate(val_loader, r, epoch)
            prec1 = 0
            if r.stage != r.num_stages: prec1 = 0

            # remember best prec@1 and save checkpoint
            best_prec1 = max(prec1, best_prec1)

            should_save_checkpoint = r.rank_in_stage == 0
            if args.checkpoint_dir and should_save_checkpoint:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': r.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict()
                }, args.checkpoint_dir, r.stage, epoch)

s = [[], []]

def schedule(buffer_list, wait_list, ops, stage):

    if stage == 1:
        if buffer_list != s[0] or wait_list != s[1]:
            s[0] = buffer_list.copy()
            s[1] = wait_list.copy()
            # print(buffer_list, wait_list)

    # buffer_list.sort()
    # wait_list.sort()

    p = [0, 15, 31, 48, 48]

    start = p[stage]
    end = p[stage + 1]

    #print(ops[0])
    #if stage == 0 :
    #    print("Buffer list ", buffer_list, "Stage list ", wait_list)

    for idx, b_val in enumerate(buffer_list):
        #op_list = ops[i][start:end]
        ok = True
        for w_val in wait_list:
            if b_val <= w_val:
                continue
        
            #print("Stage ", stage, " check buffer ", b_val, "wait ", w_val)

            for i in range(start, end):
                if ops[b_val][i] == ops[w_val][i]:
                    ok = False
                    #print("Stage ", stage, " break by buffer ", b_val, "wait ", w_val)
                    break
        if ok:
            # print("Stage ", stage, ": Scheduled with ", b_val)
            return idx, b_val
    
    return -1, -1


def train(train_loader, r, optimizer, epoch, lr_scheduler=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    n = r.num_iterations(loader_size=len(train_loader))
    if args.num_minibatches is not None:
        n = min(n, args.num_minibatches)

    # accumulation = 32
    # n -= (n % accumulation)
    # assert n % accumulation == 0

    r.train(n)
    if not is_first_stage(): train_loader = None
    r.set_loader(train_loader)

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
    for t in range(1, total_updates+1):
        
        # start num_warmup_minibatches forward passes
        # for i in range(num_warmup_minibatches):
        #     r.receive_tensors_forward()
        #     r.run_forward()

        cnt = 0
        while True:
            end = time.time()
            # perform forward pass

            seq_id = -1 
            buffer_idx = -1

            buffer_idx, seq_id = schedule(r.buffer_list, r.wait_list, r.ops, r.stage)

            #print("Stage ", r.stage, ": buffer size :", len(r.buffer_list))
            while seq_id < 0 and len(r.buffer_list) <30 and r.receive_tensors_forward():
            #while len(r.wait_list) == 0 and seq_id < 0 and len(r.buffer_list) <1 and r.receive_tensors_forward():
                # print("Stage ", r.stage, ": receive forward with :", r.tensors[-1]["seq"])
                r.buffer_list.append(r.tensors[-1]["seq"])

                cur_seq = r.tensors[-1]["seq"]
                if cur_seq >= r.wait_upper:
                    for i in range(r.wait_upper, cur_seq + 1):
                        r.wait_list.append(i)
                    r.wait_upper = cur_seq + 1

                buffer_idx, seq_id = schedule(r.buffer_list, r.wait_list, r.ops, r.stage)

                # print("seq_id", seq_id)

                if seq_id >= 0:
                    break
                
            #seq_id = self.tensors[-1][""]
            if seq_id >= 0:
                #predict next forward
                temp_buffer = r.buffer_list.copy()
                temp_buffer.pop(buffer_idx)
                temp_idx, temp_id = schedule(temp_buffer, r.wait_list, r.ops, r.stage)

                
                r.run_forward(idx=seq_id)

                r.buffer_list.pop(buffer_idx)

                            
                # print("Stage ", r.stage, ": forward ", seq_id, " buffer size ", len(r.buffer_list), flush=True)

                if is_last_stage():
                # measure accuracy and record loss
                    output, target, loss, num_tokens = r.output, r.target, r.loss.item(), r.num_tokens()
                    #print(loss, num_tokens)
                    # math.log(2) = 0.69314718  
                    # losses.update(loss / num_tokens /0.69314718, num_tokens)
                    losses.update(loss, num_tokens)

            # receive 

            if r.receive_tensors_backward():
            
                # predict
                temp_wait = r.wait_list.copy()
                temp_bwd_seq_id = r.current_bwd_idx
                for idx, val in enumerate(temp_wait):
                    if val == temp_bwd_seq_id:
                        temp_wait.pop(idx)
                        # print("Stage ", r.stage, ": Remove seq", val, " from wait_list")
                        break
                temp_idx, temp_id = schedule(r.buffer_list, temp_wait, r.ops, r.stage)

                # perform backward pass
                r.run_backward()
                # print("backward finished")
                bwd_seq_id = r.current_bwd_idx
                # print("Stage ", r.stage, ": backward ", bwd_seq_id, flush=True)

                cnt += 1

                for idx, val in enumerate(r.wait_list):
                    if val == bwd_seq_id:
                        r.wait_list.pop(idx)
                        # print("Stage ", r.stage, ": Remove seq", val, " from wait_list")
                        break

                if is_last_stage():
                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()
                    epoch_time = (end - epoch_start_time) / 3600.0
                    full_epoch_time = (epoch_time / float(t)) * float(n)

                    
                    #print("Stage ", r.stage, ": backward ", bwd_seq_id, flush=True)

                    if True:#t % args.print_freq == 0:
                        print('Stage: [{0}] Epoch: [{1}][{2}/{3}]\t'
                            'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Id: {id}\t'
                            'Tokens: {token}\t'
                            'Output: {loss.val:.32f}\t'.format(
                            args.stage, epoch, t, n, id=bwd_seq_id, token=num_tokens, timestamp=time.time(), batch_time=batch_time,
                            epoch_time=epoch_time, full_epoch_time=full_epoch_time,
                            loss=losses, # top1=top1, top5=top5,
                            memory=(float(torch.cuda.memory_allocated()) / 10**9),
                            cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
                        import sys; sys.stdout.flush()
                else:
                    if t % args.print_freq == 0:
                        # print('Stage: [{0}] Epoch: [{1}][{2}/{3}]\tMemory: {memory:.3f} ({cached_memory:.3f})'.format(
                        #     args.stage, epoch, t, n, memory=(float(torch.cuda.memory_allocated()) / 10**9),
                        #     cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
                        import sys; sys.stdout.flush()

                break
            # if i == 500 and args.local_rank == 0:
            #     subprocess.Popen(['python', 'usage.py', 'gpu.log'])

        for group in optimizer.param_groups:
            for p in group['params']:                   
                if hasattr(p, "step_"):
                    #print(p.size())
                    #print(p.name_)
                    #print(torch.sum(p.data))
                    p.grad = None

                if p.grad != None:
                    p.data = p.data.cuda()
                    if p.grad.data.get_device() != p.data.get_device():
                        print("grad ", p.grad.data.get_device()) 
                        print("data ", p.data.get_device())
                    
        optimizer.step()
        # print("Stage ", r.stage, ": step")
        r.modules_with_dependencies.modules()[0].param_cpu()
        if args.fp16:
            r.zero_grad()
        else:
            #optimizer.zero_grad()
            for group in optimizer.param_groups:
                for p in group['params']:   
                    if p.grad is not None:
                        p.grad = None
                        if not hasattr(p, "type_"):
                            p.data = p.data.cpu()
        num_updates = epoch * total_updates + t + 1
        #lr_scheduler.step_update(num_updates)

    # wait for all helper threads to complete
    r.wait()

    print("Epoch %d: %.3f seconds" % (epoch, time.time() - epoch_start_time))
    print("Epoch start time: %.3f, epoch end time: %.3f" % (epoch_start_time, time.time()))


def validate(val_loader, r, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    n = r.num_iterations(loader_size=len(val_loader))
    if args.num_minibatches is not None:
        n = min(n, args.num_minibatches)
    r.eval(n)
    if not is_first_stage(): val_loader = None
    r.set_loader(val_loader)

    end = time.time()
    epoch_start_time = time.time()

    if args.no_input_pipelining:
        num_warmup_minibatches = 0
    else:
        num_warmup_minibatches = r.num_warmup_minibatches

    if args.verbose_frequency > 0:
        print("Letting in %d warm-up minibatches" % num_warmup_minibatches)
        print("Running validation for %d minibatches" % n)

    with torch.no_grad():
        for i in range(num_warmup_minibatches):
            r.run_forward()

        for i in range(n - num_warmup_minibatches):
            # perform forward pass
            r.run_forward()
            r.run_ack()

            if is_last_stage():
                output, target, loss, num_tokens = r.output, r.target, r.loss.item(), r.num_tokens()

                # measure accuracy and record loss
                # prec1, prec5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss, output.size(0))
                # top1.update(prec1[0], output.size(0))
                # top5.update(prec5[0], output.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print('Test: [{0}][{1}/{2}]\t'
                          'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Memory: {memory:.3f} ({cached_memory:.3f})\t'
                          'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                           epoch, i, n, batch_time=batch_time, loss=losses,
                           memory=(float(torch.cuda.memory_allocated()) / 10**9),
                           cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
                    import sys; sys.stdout.flush()

        if is_last_stage():
            print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        for i in range(num_warmup_minibatches):
             r.run_ack()

        # wait for all helper threads to complete
        r.wait()

        print('Epoch %d: %.3f seconds' % (epoch, time.time() - epoch_start_time))
        print("Epoch start time: %.3f, epoch end time: %.3f" % (epoch_start_time, time.time()))

    return top1.avg


# TODO: Verify that checkpointing works correctly for GNMT
def save_checkpoint(state, checkpoint_dir, stage, epoch):
    assert os.path.isdir(checkpoint_dir)
    checkpoint_file_path = os.path.join(checkpoint_dir, "checkpoint.%d.pth.tar.epoch.%d" % (stage, epoch))
    torch.save(state, checkpoint_file_path)
    print("Saved checkpoint to %s" % checkpoint_file_path)


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
