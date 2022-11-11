# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
from glob import glob

import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from engine_finetune import evaluate_ours

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_cmae

from engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--kill_after', default=1000, type=int)

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                    help='ratio of masked-out patches.')

    parser.add_argument('--mask_ratio_eval', default=0.75, type=float,
                    help='ratio of masked-out patches in online eval.')

    parser.add_argument('--temperature', default=0.1, type=float,
                    help='temperature for softmax in InfoNCE.')

    parser.add_argument('--contextless_model', default='base', type=str, help='base / resnet')
    parser.add_argument('--contextless_model_projector_arch', type=str, default='768-768') # 2 FC layers
    parser.add_argument('--norm_vit_projector', type=int, default=0) # 2 FC layers
    parser.add_argument('--model_projector_arch', type=str, default='768-768') # 2 FC layers
    parser.add_argument('--aug_suite', default='standard', type=str, help='standard / masking')
    parser.add_argument('--wandb_log', default=None, type=str, help='all / None / gradients')

                    


    parser.add_argument('--no_wandb', action='store_false', dest='use_wandb')
    parser.set_defaults(use_wandb=True)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)
    parser.add_argument('--no_cls_token', action='store_false', dest='use_cls_token')
    parser.add_argument('--weighted_invariance', default=0., type=float)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--project_name', default='cmae',
                        help='path where to tensorboard log')  
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--linear_eval', default=1, type=int)
    parser.add_argument('--linear_eval_bn', default=1, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # losses parameters
    parser.add_argument('--detach', action='store_true', help='for pred loss, detach reps?')
    parser.add_argument('--loss_invar_coeff', type=float, default=25.)
    parser.add_argument('--loss_var_coeff', type=float, default=25.)
    parser.add_argument('--batch_patch_ratio', type=float, default=1.)
    parser.add_argument('--loss_cov_coeff', type=float, default=767.)
    parser.add_argument('--use_batch_stats', action='store_true', help='use batchwsie cov and var or only patchwise?')
    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    # seed = args.seed + misc.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    if args.aug_suite == 'standard':
        augs = [transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                transforms.RandomHorizontalFlip()]
    elif args.aug_suite == 'masking':
        augs = [transforms.Resize((args.input_size, args.input_size), interpolation=3)]
    else:
        raise ValueError("Wrong suite")
    augs.extend([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transform_train = transforms.Compose(augs)
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'),
                                         transform=transform_train)
    print(dataset_train)
    val_dataset = datasets.ImageFolder(os.path.join(args.data_path, "val"),
                                       transforms.Compose(
                                           [
                                               transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize(
                                                   mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                                               ),
                                           ]
                                       )
                                       )

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))
    if len(val_dataset) % num_tasks != 0:
        print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
              'This will slightly alter validation results as extra duplicate entries are added to achieve '
              'equal num of samples per-process.')
    sampler_val = torch.utils.data.DistributedSampler(
        val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias

    val_loader = torch.utils.data.DataLoader(
        val_dataset, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model = models_cmae.__dict__[args.model](args=args, contextless_model=args.contextless_model)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    if args.resume == '':
        fns = glob(os.path.join(args.output_dir, "checkpoint-*.pth"))
        if len(fns) > 0:
            fn = sorted(fns, key=lambda x: x.split('-')[1].split('.')[0])[-1]
            args.resume = fn

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    log_writer = None
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        if args.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=args.project_name, entity="cppr", config=vars(args))
                if args.wandb_log is not None:
                    wandb.watch(models=model_without_ddp, log=args.wandb_log, log_freq=100, log_graph=True)
                wandb.run.name = os.path.split(args.output_dir)[-1]
                wandb.run.save()
            except Exception as e:
                print(f"Unable to setup wandb: {e}")
        print(args)


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler, model_without_ddp=model_without_ddp,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        model.eval()
        stats = evaluate_ours(val_loader, model, device, args.mask_ratio_eval)
        log_stats.update(stats)

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

            if args.use_wandb:
                wandb.log(log_stats)

        if epoch > args.kill_after or (epoch == 10 and log_stats['acc1'] < 2) or (epoch == 50 and log_stats['acc1'] < 10):
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
