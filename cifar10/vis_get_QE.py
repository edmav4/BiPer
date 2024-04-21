import argparse
import os
import time
import logging
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import models_cifar
import numpy as np
from torch.autograd import Variable
from utils import *
from modules import *
from datetime import datetime
import dataset
from torch.utils.tensorboard import SummaryWriter

# checkpoint_file = './result_2/claudiaBiper_stg1_lr=0.1_rho=0.0_real/d'
# args.frequency = 40
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def main(checkpoint_file=None, ckpt_freq=None, model=None, data_path=None):
    global args, best_prec1, conv_modules
    args.data_path = data_path
    if data_path is None:
        raise ValueError('No data_path specified')
    best_prec1 = 0
    if not model is None:
        args.model = model
    # if args.evaluate:
    #     args.results_dir = '/tmp'
    # save_path = os.path.join(args.results_dir, args.save)
    # import pdb; pdb.set_trace()
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    save_path = '/tmp'

    if not args.resume and not args.evaluate:
        with open(os.path.join(save_path, 'config.txt'), 'w') as args_file:
            args_file.write(str(datetime.now()) + '\n\n')
            for args_n, args_v in args.__dict__.items():
                args_v = '' if not args_v and not isinstance(args_v, int) else args_v
                args_file.write(str(args_n) + ':  ' + str(args_v) + '\n')

        setup_logging(os.path.join(save_path, 'logger.log'))
        logging.debug("run arguments: %s", args)
    else:
        setup_logging(os.path.join(save_path, 'logger.log'), filemode='a')

    if 'cuda' in args.type:
        args.gpus = [0]
        if args.seed > 0:
            set_seed(args.seed)
        else:
            cudnn.benchmark = True
    else:
        args.gpus = None

    if args.dataset == 'tinyimagenet':
        num_classes = 200
        model_zoo = 'models_imagenet.'
    elif args.dataset == 'imagenet':
        num_classes = 1000
        model_zoo = 'models_imagenet.'
    elif args.dataset == 'cifar10':
        num_classes = 10
        model_zoo = 'models_cifar.'
    elif args.dataset == 'cifar100':
        num_classes = 100
        model_zoo = 'models_cifar.'

    # * create model
    if len(args.gpus) == 1:
        model = eval(model_zoo + args.model)(num_classes=num_classes).cuda()
    else:
        model = nn.DataParallel(eval(model_zoo + args.model)(num_classes=num_classes))
    if not args.resume:
        logging.info("creating model %s", args.model)
        # logging.info("model structure: ")
        # for name, module in model._modules.items():
        #     logging.info('\t' + str(name) + ': ' + str(module))
        # num_parameters = sum([l.nelement() for l in model.parameters()])
        # logging.info("number of parameters: %d", num_parameters)

    ''' Load checkpoint '''
    # if checkpoint_file is None raise error
    if checkpoint_file is None:
        raise ValueError('No checkpoint_file specified')

    # print(checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
    if len(args.gpus) > 1:
        checkpoint['state_dict'] = dataset.add_module_fromdict(checkpoint['state_dict'])
    checkpoint['epoch'] = 1
    args.start_epoch = checkpoint['epoch'] - 1
    model.load_state_dict(checkpoint['state_dict'])
    logging.info("loaded checkpoint '%s' (epoch %s) successfully",
                 checkpoint_file, checkpoint['epoch'])

    logging.info("Compuing QE and b ...")

    model = model.type(args.type)

    # * load dataset
    train_loader, val_loader = dataset.load_data(
        dataset=args.dataset,
        data_path=args.data_path,
        batch_size=args.batch_size,
        batch_size_test=args.batch_size_test,
        num_workers=args.workers)

    # * optimizer settings
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': args.lr}],
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': args.lr}],
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        logging.error("Optimizer '%s' not defined.", args.optimizer)

    if args.lr_type == 'cos':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - args.warm_up * 4, eta_min=0,
                                                                  last_epoch=args.start_epoch)
    elif args.lr_type == 'step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_decay_step, gamma=0.1, last_epoch=-1)
    elif args.lr_type == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (
                    1.0 - (epoch - args.warm_up * 4) / (args.epochs - args.warm_up * 4)), last_epoch=-1)

    ''' Evaluation QE and b'''
    # * record names of conv_modules
    conv_modules = []
    for name, module in model.named_modules():
        if isinstance(module, BinarizeConv2d):
            conv_modules.append(module)

    ''' Quantization error'''
    args.frequency = ckpt_freq
    QE = 0
    b = 0
    for module in model.modules():
        if 'Binarize' in module._get_name():
            bw = torch.sin(args.frequency * module.weight)
            # bw = module.weight
            alpha = module.alpha[:, :, :, None]
            # import pdb; pdb.set_trace()
            QE = QE + torch.mean(torch.abs(bw - torch.sign(bw)))
            b = b + torch.mean(torch.abs(module.weight))
    print(f'QE={QE}')
    print(f'b={b}')
    return QE, b

if __name__ == '__main__':
    main()