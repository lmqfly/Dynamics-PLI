import torch

import os
import logging
import argparse
import random
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Build and train Molformer.")
    # set pretraining hyper-parameters
    parser.add_argument('--ordering', type=bool, default=False, help='Whether to use the snapshot ordering task.')
    parser.add_argument('--noise', type=bool, default=False, help='Whether to add noise during pretraining.')
    parser.add_argument('--prompt', type=str, default='1', help='Whether to use the snapshot ordering task.')
    parser.add_argument('--max_sample', type=int, default=3250, help='The default number of pre-train samples.')

    # set fine-tune hyper-parameters
    parser.add_argument('--data', type=str, default='lba', choices=['lba', 'lep'])
    parser.add_argument('--unknown', type=bool, default=False, help='Whether to use docking structures.')
    parser.add_argument('--pre', type=bool, default=True, help='Whether to use docking structures.')
    parser.add_argument('--pretrain', type=str, default='', help='Whether to load the pretrained model weights.')
    parser.add_argument('--linear_probe', type=int, default=0, help='linear_probe.')
    parser.add_argument('--epochs_finetune', type=int, default=30, help='Number of epoch.')

    # set model hyper-parameters
    parser.add_argument('--dim', type=int, default=128, help='Dimension of features.')
    parser.add_argument('--dropout', type=float, default=0.15, help='Dropout rate.')
    parser.add_argument('--depth_local', type=int, default=6, help='Number of local layers.')
    parser.add_argument('--depth_cross', type=int, default=2, help='Number of cross layers.')
    parser.add_argument('--cross_cutoff', type=int, default=9, help='cutoff of local.')
    parser.add_argument('--local_cutoff', type=int, default=4, help='cutoff of cross.')

    # set training details
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--split_ratio', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=60, help='Number of epoch.')
    parser.add_argument('--bs', type=int, default=4, help='Batch size.')
    parser.add_argument('--atoms_bacth', type=int, default=10000, help='atoms in single batch.')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate.')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='minimum learning rate.')
    parser.add_argument('--resume', type=bool, default=False, help='Training from checkpoint')

    # set training environment
    parser.add_argument('--split', type=str, default='30', choices=['30', '60'])
    parser.add_argument('--data_dir', type=str, default='../data_npy_refine/', help='Path for loading data.')
    parser.add_argument('--gpu', type=str, default='0', help='Index for GPU')
    # parser.add_argument("--local_rank", default= -1)
    parser.add_argument('--save_path', default='save/', help='Path to save the model and the logger.')



    args = parser.parse_args()
    return args
    
def set_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    random.seed(s)

    # keep cudnn stable for CNN，https://learnopencv.com/ensuring-training-reproducibility-in-pytorch/
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger(object):
    level_relations = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING,
                       'error': logging.ERROR, 'crit': logging.CRITICAL}  # 日志级别关系映射

    def __init__(self, path, filename, level='info'):
        if not os.path.exists(path):
            os.makedirs(path)
        self.logger = logging.getLogger(path + filename)
        self.logger.setLevel(self.level_relations.get(level))

        sh = logging.StreamHandler()
        self.logger.addHandler(sh)

        th = logging.FileHandler(path + filename, encoding='utf-8')
        self.logger.addHandler(th)

def Batchs_with_constant_atoms(trainset,batch_atom_nums):
    # prepare batchs ensures atom nums of batchs less than predefined threshhold
    Batchs = []
    data_batch = []
    tmp = 0
    for idx,d in enumerate(trainset):
        tmp += (torch.logical_and(trainset[idx].x > 1,trainset[idx].x!=98)).sum()
        data_batch.append(idx)
        if (idx < len(trainset) -1) and (torch.logical_and(trainset[idx].x > 1,trainset[idx].x!=98).sum() + tmp < batch_atom_nums):
            continue
        else:
            Batchs.append(data_batch)
            tmp = 0
            data_batch = []
    return Batchs

if __name__ == '__main__':
    print()
