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
import torch.nn.functional as F
import util.misc as misc
import util.lr_sched as lr_sched
from models_cmae import LossLog


def train_one_epoch(encoder: torch.nn.Module,
                    predictor: torch.nn.Module,
                    discriminator: torch.nn.Module,
                    lin_prob_model: torch.nn.Module,
                    data_loader: Iterable,
                    gen_optimizer: torch.optim.Optimizer,
                    disc_optimizer: torch.optim.Optimizer,
                    lin_prob_optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):

    models = [encoder, predictor, discriminator, lin_prob_model]
    optimizers = [gen_optimizer, disc_optimizer, lin_prob_optimizer]

    for model in models:
        model.train(True)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    for optimizer in optimizers:
        optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, y) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            for optimizer in optimizers:
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        loss_log = LossLog()

        with torch.cuda.amp.autocast():

            # real examples
            with torch.no_grad():
                real_full_reps, _ = encoder(samples, mask_ratio=None).detach()

            # generated examples
            masked_reps, ids_restore = encoder(samples, mask_ratio=args.mask_ratio)
            pred_full_reps = predictor(masked_reps, ids_restore)

            # stack reals and fakes, apply discriminator
            disc_input = torch.cat(real_full_reps, pred_full_reps)
            disc_output = discriminator(disc_input)

            # discriminator loss
            disc_labels = torch.cat([torch.ones_like(real_full_reps),
                                     torch.zeros_like(pred_full_reps)], 0)

            loss_disc = F.mse_loss(disc_output, disc_labels)
            loss_disc = loss_log.add_loss('loss_disc', 1., loss_disc)

            # generator loss
            disc_output_for_gen = disc_output[real_full_reps.shape[0]:]
            gen_labels = torch.ones_like(pred_full_reps)
            gen_loss = F.mse_loss(disc_output_for_gen, gen_labels)
            gen_loss = loss_log.add_loss('loss_gen', 1., gen_loss)

            # online liniar probing eval loss
            loss_lin_prob = lin_prob_model(real_full_reps, y)
            loss_lin_prob = loss_log.add_loss('loss_lin_prob', 1., loss_lin_prob)

        loss_gen_value = gen_loss.item()
        loss_disc_value = loss_disc.item()
        loss_lin_prob_value = loss_disc.item()

        loss_dict = loss_log.return_loss()
        loss_val = loss_dict['loss']
        if not math.isfinite(loss_val.item()):
            print("Loss is {}, stopping training".format(loss_val.item()))
            sys.exit(1)

        loss_gen_value /= accum_iter
        loss_scaler(gen_loss, gen_optimizer, parameters=list(encoder.parameters())+list(predictor.parameters()),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        loss_disc_value /= accum_iter
        loss_scaler(loss_disc, disc_optimizer, parameters=discriminator.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        loss_lin_prob_value /= accum_iter
        loss_scaler(loss_lin_prob, lin_prob_optimizer, parameters=lin_prob_model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            gen_optimizer.zero_grad()
            disc_optimizer.zero_grad()
            lin_prob_optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(lr_gen=gen_optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_disc=disc_optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_lin_prob=lin_prob_optimizer.param_groups[0]["lr"])
        metric_logger.update(**loss_dict)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
