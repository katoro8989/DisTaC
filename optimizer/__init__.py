# coding: utf-8
import torch.optim as optim
from .sam import SAM
import math


def build_optimizer(args, params):
    name = args.optimizer
    model_params = params

    # Standard Optimizer
    if name == 'vanilla_sgd':
        return optim.SGD(params=model_params, 
                        lr=args.lr, 
                        weight_decay=args.wd)

    elif name == 'momentum_sgd':
        return optim.SGD(params=model_params, 
                        lr=args.lr, 
                        momentum=args.momentum,
                        weight_decay=args.wd)

    elif name == 'adam':
        return optim.Adam(params=model_params, 
                        lr=args.lr, 
                        betas=(args.beta_1, args.beta_2), 
                        eps=args.eps, 
                        weight_decay=args.wd, 
                        amsgrad=False)

    elif name == 'adamw':
        return optim.AdamW(params=model_params, 
                        lr=args.lr, 
                        betas=(args.beta_1, args.beta_2), 
                        eps=args.eps, 
                        weight_decay=args.wd, 
                        amsgrad=False)

    elif name == 'sam':
        return SAM(params=model_params, 
                   base_optimizer=optim.SGD,
                   rho=args.rho,
                   eps=args.eps,
                   lr=args.lr, 
                   weight_decay=args.wd, 
                   momentum=args.momentum)

    else:
        raise ValueError(
            'The selected optimizer is not supported for this trainer.')

      
def linear_warmup_cosine_decay(warmup_steps, total_steps):

    def fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn