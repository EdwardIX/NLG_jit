#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import math
import os
from collections import OrderedDict 
import argparse

import jittor as jt
from jittor.optim import Optimizer, LambdaLR


def add_optimizer_params(parser: argparse.ArgumentParser):
    parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay rate')
    parser.add_argument('--correct_bias', action='store_true', help='correct adam bias term')
    parser.add_argument('--adam_epislon', default=1e-6, type=float, help='adam epsilon')
    parser.add_argument('--no_decay_bias', action='store_true', help='no weight decay on bias weigh')
    parser.add_argument('--adam_beta1', default=0.9, type=float, help='adam beta1 term')
    parser.add_argument('--adam_beta2', default=0.98, type=float, help='adam beta2 term')
    
    parser.add_argument('--scheduler', default='linear', type=str,
                        choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant', 'linear', 'cycle'],
                        help='lr scheduler to use.')

    parser.add_argument('--max_step', type=int, default=None, help='upper epoch limit')

    parser.add_argument('--max_epoch', type=int, default=None, help='max epoch of training')

    parser.add_argument('--warmup_step', type=int, default=0, help='upper epoch limit')

    parser.add_argument('--i_steps', type=str, default='0', help='interval_steps')
    parser.add_argument('--i_lrs', type=str, default='0.00025', help='interval_lrs')


class AdamW(Optimizer):
    """ AdamW Optimizer. (Adapted from source code of jittor by suc)
    
    Example::

        optimizer = nn.AdamW(model.parameters(), lr, eps=1e-8, betas=(0.9, 0.999))
        optimizer.step(loss)
    """
    def __init__(self, params, lr, eps=1e-8, betas=(0.9, 0.999), weight_decay=0):
        super().__init__(params, lr)
        self.eps = eps
        self.betas = betas
        self.weight_decay = weight_decay
        # assert weight_decay==0, "weight_decay is not supported yet"
        
        # initialize required arguments for each param_groups
        for pg in self.param_groups:
            values = pg["values"] = []
            m = pg["m"] = []
            for p in pg["params"]:
                values.append(jt.zeros(p.shape, p.dtype).stop_grad())
                m.append(jt.zeros(p.shape, p.dtype).stop_grad())

    def add_param_group(self, group):
        values = group["values"] = []
        m = group["m"] = []
        for p in group["params"]:
            values.append(jt.zeros(p.shape, p.dtype).stop_grad())
            m.append(jt.zeros(p.shape, p.dtype).stop_grad())
        self.param_groups.append(group)

    def step(self, loss=None, retain_graph=False):
        self.pre_step(loss, retain_graph)
        n = float(self.n_step)
        for pg in self.param_groups:
            # get arguments from each param_groups
            lr = pg.get("lr", self.lr)
            eps = pg.get("eps", self.eps)
            weight_decay = pg.get("weight_decay", self.weight_decay)
            b0, b1 = pg.get("betas", self.betas)
            for p, g, v, m in zip(pg["params"], pg["grads"], pg["values"], pg["m"]):
                if p.is_stop_grad(): continue
                bias_correction1 = 1 - b0 ** n
                bias_correction2 = 1 - b1 ** n
                m.update(b0 * m + (1-b0) * g) #exp_avg
                v.update(b1 * v + (1-b1) * g * g) #exp_avg_sq
                denom = jt.sqrt(v) / jt.sqrt(bias_correction2) + eps
                step_size = lr / bias_correction1
                p.update(p - step_size * m / denom)
                p.update(p * (1 - lr * weight_decay)) # Put the weight decay at the end (fixed version)
        self.post_step()


def get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    last_epoch=-1
):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def create_grouped_parameters(model, no_decay_bias): # args):
    if not no_decay_bias:
        optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()], # if not any(nd in n for nd in no_decay)],
        }]
    else:
        no_decay = ["bias", "layer_norm.weight"]

        optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
            "weight_decay": 0.0,
        }]
    return optimizer_grouped_parameters


def create_adam_optimizer_from_args(model, args, grouped_parameters=None):
    if grouped_parameters is None:
        grouped_parameters = create_grouped_parameters(model, args.no_decay_bias)

    optimizer = AdamW(
        grouped_parameters, 
        lr=args.lr, 
        betas=(args.adam_beta1, args.adam_beta2), 
        eps=args.adam_epislon, 
        weight_decay=args.weight_decay, 
        # correct_bias=args.correct_bias
    )
    assert args.correct_bias 
    return optimizer


def create_optimizer_scheduler(optimizer, args):
    if args.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, args.warmup_step, args.max_step, last_epoch=-1
        )
    else:
        raise ValueError('Scheduler currently only support linear')
    return scheduler
