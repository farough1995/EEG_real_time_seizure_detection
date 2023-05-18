
import numpy as np
import os
import argparse
import random
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from itertools import groupby
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable

from torchinfo import summary

from builder.utils.lars import LARC
from control.config import args

from builder.data.data_preprocess import get_data_preprocessed
# from builder.data.data_preprocess_temp1 import get_data_preprocessed
from builder.models import get_detector_model, grad_cam
from builder.utils.logger import Logger
from builder.utils.utils import set_seeds, set_devices
from builder.utils.cosine_annealing_with_warmup import CosineAnnealingWarmUpRestarts
from builder.utils.cosine_annealing_with_warmupSingle import CosineAnnealingWarmUpSingle
from builder.utils.result_utils import experiment_results_validation , experiment_results
from builder.trainer import get_trainer
from builder.trainer import *


from torch.profiler import profile, record_function, ProfilerActivity
from prettytable import PrettyTable
import pandas as pd

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(table)
    print(f"Total Params: {pytorch_total_params}")
    print(f"Total Trainable Params: {total_params}")
    

def my_profiler(model, inputs):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True, record_shapes=True) as prof:
        with record_function("model_inference"):
            model(inputs)

    return prof.key_averages().table(row_limit=100)
    
def summary_torchinfo(model , loader, save_path):
    x_batch= next(iter(loader))
    batch = x_batch[0]
    with open(os.path.join(save_path, 'torchinfo.txt'), 'a') as f:
                f.write(str(summary(model, input_data=batch)))
                f.write("\n")
                f.close()
    
    
    

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
list_of_test_results_per_seed = []



dir_root = os.path.join(args.dir_result, args.project_name)
dir_pro = os.path.join(dir_root, 'profile')
if not os.path.exists(dir_pro):
    os.makedirs(dir_pro)

for seed_num in args.seed_list:
    args.seed = seed_num
    set_seeds(args)
    device = set_devices(args)

    # Load Data, Create Model 
    train_loader, val_loader, test_loader, len_train_dir, len_val_dir, len_test_dir = get_data_preprocessed(args)
    model = get_detector_model(args) 
    val_per_epochs = 10
    model = model(args, device).to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')

    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr = args.lr_init, weight_decay=args.weight_decay)
    elif args.optim == 'adam_lars':
        optimizer = optim.Adam(model.parameters(), lr = args.lr_init, weight_decay=args.weight_decay)
        optimizer = LARC(optimizer=optimizer, eps=1e-8, trust_coefficient=0.001)
    elif args.optim == 'sgd_lars':
        optimizer = optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer = LARC(optimizer=optimizer, eps=1e-8, trust_coefficient=0.001)
    elif args.optim == 'adamw_lars':
        optimizer = optim.AdamW(model.parameters(), lr = args.lr_init, weight_decay=args.weight_decay)
        optimizer = LARC(optimizer=optimizer, eps=1e-8, trust_coefficient=0.001)

    one_epoch_iter_num = len(train_loader)
    print("Iterations per epoch: ", one_epoch_iter_num)
    iteration_num = args.epochs * one_epoch_iter_num

    if args.lr_scheduler == "CosineAnnealing":
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=args.t_0*one_epoch_iter_num, T_mult=args.t_mult, eta_max=args.lr_max, T_up=args.t_up*one_epoch_iter_num, gamma=args.gamma)
    elif args.lr_scheduler == "Single":
        scheduler = CosineAnnealingWarmUpSingle(optimizer, max_lr=args.lr_init * math.sqrt(args.batch_size), epochs=args.epochs, steps_per_epoch=one_epoch_iter_num, div_factor=math.sqrt(args.batch_size))

    
    model = get_detector_model(args) 
    val_per_epochs = 2

    print("#################################################")
    print("################# Profiler Begins ###################")
    print("#################################################")
    model = model(args, device).to(device)
    logger = Logger(args)
    # load model checkpoint  
    if args.last:
        ckpt_path = args.dir_result + '/' + args.project_name + '/ckpts/last_{}.pth'.format(str(seed_num))
    elif args.best:
        ckpt_path = args.dir_result + '/' + args.project_name + '/ckpts/best_{}.pth'.format(str(seed_num))

    if not os.path.exists(ckpt_path):
        print("Final model for test experiment doesn't exist...",ckpt_path)
        exit(1)
    # load model & state
    ckpt    = torch.load(ckpt_path, map_location=device)
    state   = {k: v for k, v in ckpt['model'].items()}
    model.load_state_dict(state)

    
    # initialize test step
    model.eval()
    logger.evaluator.reset()
    iteration = 0
    result_pro = []
    with torch.no_grad():
         for test_batch in tqdm(test_loader, total=len(test_loader), bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            test_x, test_y, seq_lengths, target_lengths, aug_list, signal_name_list = test_batch
            test_x, test_y = test_x.to(device), test_y.to(device)
            
            result_pro.append(my_profiler(model=model , inputs=test_x))
    
for item in result_pro:
    print(item)

# torchinfo summary
print("################# Torchinfo Begins ###################")
summary_torchinfo(model, test_loader, dir_pro)

# print("################# count params Begins ###################")
# count_parameters(model)