# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        samples = samples.to(device, non_blocking=True).view(-1, 3, args.input_size, args.input_size)
        with torch.cuda.amp.autocast():
            loss, _, _, loss_batchwise, loss_patchwise, loss_cls = model(samples, num_groups=args.num_groups, group_sz=args.group_sz)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_batchwise=loss_batchwise.item())
        metric_logger.update(loss_patchwise=loss_patchwise.item())
        metric_logger.update(loss_cls=loss_cls.item())
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def nn_evaluate(data_loader, model, device, num_queries=5, num_neighbors=5):
    # assumes use_cls_token=True and data_loader(shuffle=False)

    # switch to evaluation mode
    model.eval()
    
    query_ims = [data_loader.dataset[ind][0] for ind in range(num_queries)]

    with torch.cuda.amp.autocast():
        q_reps_full = model(query_ims, num_groups=args.num_groups, group_sz=args.group_sz, encode_only=True)
        q_reps_cls = q_reps_full[:, :args.num_groups, :].mean(1)  # [Q, E]
        queries = F.normalize(q_reps_cls, dim=-1)

    full_sim_matrix = torch.empty(num_queries, 0, device=torch.device('cpu'))

    for samples, _ in data_loader:
        # get a batch
        samples = samples.to(device, non_blocking=True)
        
        # calcualte representations
        with torch.cuda.amp.autocast():
            representations_full = model(samples, num_groups=args.num_groups, group_sz=args.group_sz, encode_only=True)
            rep_cls = representations_full[:, :args.num_groups, :].mean(1)  # [B, E]
            reps = F.normalize(rep_cls, dim=-1)

            cur_sim_matrix = torch.einsum('QE,BE->QB', pred, representations)
            
            full_sim_matrix = torch.cat((full_sim_matrix, cur_sim_matrix.to(full_sim_matrix.device)), 1)

    vals, inds = torch.topk(full_sim_matrix, num_neighbors)
    inds.view(-1)
    neighbors = data_loader.dataset[inds][0]
    full_reps = model(torch.cat([queries, neighbors]), num_groups=args.num_groups, group_sz=args.group_sz, encode_only=True)
    q_full_reps = full_reps[:num_queries]
    full_reps = full_reps[num_queries:]

    return queries, neighbors, vals