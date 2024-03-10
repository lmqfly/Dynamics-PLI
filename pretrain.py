import os
import copy
import random
from time import time, strftime, localtime
from datasets.pyg.masito import MASITO
from datasets.pyg.lep import LEPDataset
import torch
import torch.optim as opt
from scipy import stats
import nets
from nets import model_entrypoint
from tqdm import tqdm
import torch_geometric
from util_mine import parse_args, Logger, set_seed, Batchs_with_constant_atoms
from torch_geometric.data import Batch
from Model.EHGNN import EHGNN

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def run_eval(args, model, loader, length):
    model.eval()
    metric = 0
    criterion1 = torch.nn.MSELoss()
    criterion2 = torch.nn.MSELoss()
    criterion3 = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
         for batch_idx, batchs in enumerate(tqdm(loader)):
                out = model(batchs)
                metric += (criterion1(out[0].detach().cpu(),out[1].detach().cpu()) 
                           # + criterion2(out[2].detach().cpu(),out[3].detach().cpu())
                           + criterion3(out[5].detach().cpu(),out[4].detach().cpu()) * 100) * len(batchs)  / length
    return metric


def main():
    args = parse_args()
    set_seed(args.seed)
    log = Logger(f'{args.save_path}pretrain/', f'pretrain_{strftime("%Y-%m-%d_%H-%M-%S", localtime())}.log')
    
    dataset = MASITO('datasets/MASITO')
    index = [i for i in range(len(dataset))] 
    random.shuffle(index)
    dataset = dataset[index]
    
    trainset = dataset[:1047700]
    valset = dataset[1047700:]

    trainloader = torch_geometric.loader.DataListLoader(trainset, batch_size=96, shuffle=True)
    valloader = torch_geometric.loader.DataListLoader(valset, batch_size=96, shuffle=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    model = EHGNN(n_atom_basis = args.dim,
            local_cutoff = args.local_cutoff,
            cross_cutoff = args.cross_cutoff,
            layers_local = args.depth_local,
            layers_cross = args.depth_cross, 
            dropout = args.dropout,   
            head_type=0).cuda()
    
    if len(args.gpu) > 1:  model = torch_geometric.nn.DataParallel(model)
    
    criterion1 = torch.nn.MSELoss()
    criterion2 = torch.nn.MSELoss()
    criterion3 = torch.nn.CrossEntropyLoss()
    best_metric = 1e9
    
    optimizer = opt.Adam(model.parameters(), lr = args.lr,weight_decay=5e-3)
    lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=5, min_lr=args.min_lr)
    log.logger.info(f'{"=" * 40} Pretrain {"=" * 40}\n' f'Train: {len(trainset)}; Val: {len(valset)};' f'\n{"=" * 40} Start Training {"=" * 40}'
                    f'\nEmbed_dim: {args.dim};  dropout: {args.dropout}; depth_local: {args.depth_local}; depth_cross: {args.depth_cross}; local_cutoff: {args.local_cutoff};cross_cutoff: {args.cross_cutoff};'
                    f'\nSeed: {args.seed};atoms_bacth: {args.atoms_bacth};epochs: {args.epochs};lr: {args.lr};\n{"=" * 40} Start Training {"=" * 40}')
    best_model = None
    best_epoch = -1
    t0 = time()
    try:
        for epoch in range(0,args.epochs):
            model.train()
            loss = 0.0
            loss1 = 0.0
            loss2 = 0.0
            loss3 = 0.0
            t1 = time()
            for batch_idx, batchs in enumerate(tqdm(trainloader)):
                    out = model(batchs)
                    loss_batch1 = criterion1(out[1], out[0])
                    loss_batch3 = criterion3(out[5], out[4]) * 100
                    loss_batch = loss_batch1 + loss_batch3
                    loss += loss_batch.item() * len(batchs) / len(trainset)
                    loss1 += loss_batch1.item() * len(batchs) / len(trainset)
                    loss3 += loss_batch3.item() * len(batchs) / len(trainset)
                    if batch_idx % 100 ==0:
                        print(loss_batch,loss_batch1,loss_batch3)
                    loss_batch.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            metric = run_eval(args, model, valloader, len(valset))
            log.logger.info('Epoch: {} | Time: {:.1f}s | Loss1: {:.5f} | Loss3: {:.5f} | Loss: {:.5f} | Metric: {:.3f} '
                            '| Lr: {:.6f}'.format(epoch + 1, time() - t1, loss1, loss3, loss, metric, optimizer.param_groups[0]['lr']))
            lr_scheduler.step(metric)
            if metric < best_metric:
                best_metric = metric
                best_epoch = epoch + 1
                checkpoint = {'epochs': args.epochs}
                if len(args.gpu) > 1:
                    checkpoint['model'] = model.module.state_dict()
                else:
                    checkpoint['model'] = model.state_dict()
                torch.save(checkpoint, args.save_path + f'Pretrain.pt')
    except:
        log.logger.info('Training is interrupted.')
    log.logger.info('{} End Training (Time: {:.2f}h) {}'.format("=" * 20, (time() - t0) / 3600, "=" * 20))
    log.logger.info(f'Save the best model as Pretrain.pt.\nBest Epoch: {best_epoch} | '
                    f'Metric: {best_metric}')

if __name__ == '__main__':
    main()

