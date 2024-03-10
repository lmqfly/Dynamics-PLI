import os
import copy
import random
from tqdm import tqdm 
from time import time, strftime, localtime
from datasets.pyg.pdbbind import PdbbindDataset
from datasets.pyg.lep import LEPDataset
import torch
from torch_geometric.data import Batch
import torch.optim as opt
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from util_mine import parse_args, Logger, set_seed, Batchs_with_constant_atoms
from torch_geometric.data import Batch
from Model.EHGNN import EHGNN


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.set_default_dtype(torch.float32)

def run_eval(args, model, trainset,Batchs, length):
    model.eval()
    metric = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch in Batchs:
            out = model(Batch.from_data_list(trainset[batch[0]:batch[-1]+1]).cuda())
            y_pred.append(torch.sigmoid(out[1].detach().cpu()))
            y_true.append(out[0].detach().cpu())

    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    
    auroc = roc_auc_score(y_true.numpy(), y_pred.numpy())
    auprc = average_precision_score(y_true.numpy(), y_pred.numpy())
    
    return auroc, auprc, metric


def main():
    args = parse_args()
    set_seed(args.seed)
    log = Logger(f'{args.save_path}lep/', f'lep_{strftime("%Y-%m-%d_%H-%M-%S", localtime())}.log')
    
    trainset = list(LEPDataset('datasets/lep/train'))
    random.shuffle(trainset) 
    valset = LEPDataset('datasets/lep/val')
    testset = LEPDataset('datasets/lep/test')
    
    Batchs_train = Batchs_with_constant_atoms(trainset,args.atoms_bacth)
    Batchs_val = Batchs_with_constant_atoms(valset,args.atoms_bacth)
    Batchs_test = Batchs_with_constant_atoms(testset,args.atoms_bacth)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    model = EHGNN(n_atom_basis = args.dim,
                local_cutoff = args.local_cutoff,
                cross_cutoff = args.cross_cutoff,
                layers_local = args.depth_local,
                layers_cross = args.depth_cross, 
                dropout = args.dropout,   
                head_type=2).cuda()
     
    if args.pretrain:
        checkpoint = torch.load(args.save_path + args.pretrain)
        pretrained_dict = {key: value for key, value in checkpoint['model'].items() if 'head' not in key}
        pretrained_dict['local_model.res_embedding.weight'] = torch.randn(100,args.dim)
        model.load_state_dict(pretrained_dict,strict=False)
        if args.linear_probe:
            for name, param in model.named_parameters():
                if param.requires_grad:
                        if name in pretrained_dict:
                            param.requires_grad = False  

        print("Load model successfully!")
    else:
        args.pretrain = 'no_pre'
        
    if len(args.gpu) > 1:  model = torch.nn.DataParallel(model)
    criterion = torch.nn.BCELoss()
    best_metric = 0
    optimizer = opt.Adam(model.parameters(), lr = args.lr,weight_decay=1e-2)
    lambda1 = lambda cur_iter: 1 
    lr_scheduler = opt.lr_scheduler.LambdaLR(optimizer, lambda1)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    log.logger.info(f'{"=" * 40} PDBbind {"=" * 40}\n'
                    f'Train: {len(trainset)}; Val: {len(valset)}; Test: {len(testset)}; Pre-train Model: {args.pretrain}'
                    f'\nEmbed_dim: {args.dim};  dropout: {args.dropout}; depth_local: {args.depth_local}; depth_cross: {args.depth_cross}; local_cutoff: {args.local_cutoff};cross_cutoff: {args.cross_cutoff};'
                    f'\nSeed: {args.seed};atoms_bacth: {args.atoms_bacth};epochs: {args.epochs};lr: {args.lr};\n{"=" * 40} Start Training {"=" * 40}')
    best_model = None
    t0 = time()
    try:
        for epoch in range(0,args.epochs):
            model.train()
            loss = 0.0
            t1 = time()
            start_id = 0
            for batch_id,batch in enumerate(tqdm(Batchs_train)):
                out = model(Batch.from_data_list(trainset[batch[0]:batch[-1]+1]).cuda())
                loss_batch = criterion(torch.sigmoid(out[1]), (1-out[0].float())*0.1 + out[0].float()*0.9)
                loss += loss_batch.item() * len(batch) / len(trainset)
                loss_batch.backward()
                if batch_id - start_id + 1 == 8:
                    optimizer.step()
                    optimizer.zero_grad()
                    start_id = batch_id + 1
            auroc, auprc, _ = run_eval(args, model, testset, Batchs_test,len(testset))
            metric = auroc
            log.logger.info('Epoch: {} | Time: {:.1f}s | Loss: {:.2f} | AUROC: {:.3f} | AUPRC: {:.3f} '
                                '| Lr: {:.6f}'.format(epoch + 1, time() - t1, loss * 1e4, auroc, auprc, optimizer.param_groups[0]['lr']))
            auroc, auprc, _ = run_eval(args, model, valset, Batchs_val,len(valset))
            metric = auroc
            log.logger.info('Epoch: {} | Time: {:.1f}s | Loss: {:.2f} | AUROC: {:.3f} | AUPRC: {:.3f} '
                                '| Lr: {:.6f}'.format(epoch + 1, time() - t1, loss * 1e4, auroc, auprc, optimizer.param_groups[0]['lr']))
            if metric > best_metric:
                best_metric = metric
                best_model = copy.deepcopy(model)  # deep copy model
                best_epoch = epoch + 1
            lr_scheduler.step()
    except:
        log.logger.info('Training is interrupted.')
        log.logger.info('{} End Training (Time: {:.2f}h) {}'.format("=" * 20, (time() - t0) / 3600, "=" * 20))
        
    checkpoint = {'epochs': args.epochs}


    auroc, auprc, _ = run_eval(args, best_model, testset, Batchs_test,len(testset))
    if len(args.gpu) > 1:
        checkpoint['model'] = best_model.module.state_dict()
    else:
        checkpoint['model'] = best_model.state_dict()
    if args.linear_probe: args.linear_probe = 'Linear'

    torch.save(checkpoint, args.save_path + f'LEP_{args.split}_{args.pretrain}_{args.linear_probe}.pt')
    log.logger.info(f'Save the best model as LEP_{args.split}_{args.pretrain}_{args.linear_probe}.pt.\n'
                    f'Best Epoch: {best_epoch} | Test AUROC: {auroc} | Test AUPRC: {auprc}')


if __name__ == '__main__':
    main()

