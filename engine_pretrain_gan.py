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

def train_one_epoch(encoder: torch.nn.Module,
                    predictor: torch.nn.Module,
                    discriminator: torch.nn.Module,
                    lin_probe_model: torch.nn.Module,
                    data_loader: Iterable, 
                    gen_optimizer: torch.optim.Optimizer,
                    disc_optimizer: torch.optim.Optimizer,
                    lin_probe_optimizer: torch.optim.Optimizer,
                    criterion: torch.nn.Module,
                    device: torch.device, epoch: int, loss_scaler, model_without_ddp=None,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    gen_optimizer.zero_grad()
    disc_optimizer.zero_grad()
    if lin_probe_model is not None:
        lin_probe_optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, y) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        samples = samples.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)d
        with torch.cuda.amp.autocast():
            
            # real examples
            with torch.no_grad():
                real_full_reps, _ = encoder(x, mask_ratio=None).detach()

            # generated examples
            masked_reps, ids_restore = encoder(x, mask_ratio=args.mask_ratio)
            pred_full_reps = predictor(masked_reps, ids_restore)

            # stack reals and fakes, apply discriminator
            disc_input = torch.cat(real_full_reps, pred_full_reps)
            disc_output = discriminator(disc_input)

            # discriminator loss
            disc_labels = torch.cat([torch.ones_like(real_full_reps), 
                                     torch.zeros_like(pred_full_reps)], 0)
            loss_disc = criterion(disc_output, disc_labels)
            loss_log.add_loss('loss_disc', 1., loss_disc)
            
            # generator loss
            disc_output_for_gen = disc_output[real_full_reps.shape[0]:]
            gen_labels = torch.zeros_like(pred_full_reps)
            gen_loss = criterion(disc_output_for_gen, gen_labels)
            loss_log.add_loss('loss_gen', 1., loss_gen)

            # online liniar probing eval loss
            if lin_prob_model is not None:
                loss_lin_prob = lin_prob_model(real_full_reps, y)
                loss_log.add_loss('loss_lin_prob', 1., loss_lin_prob)

        loss = loss_dict['loss']
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
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(**loss_dict)
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
