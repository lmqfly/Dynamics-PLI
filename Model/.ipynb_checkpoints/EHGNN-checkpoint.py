from .utils import GaussianRBF,CosineCutoff
from .PaiNN import PaiNN
from .BiMP import BiMP
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from .utils import Dense
class EHGNN(nn.Module):
    def __init__(self,
                n_atom_basis = 128,
                local_cutoff = 10,
                cross_cutoff = 20,
                layers_local = 6,
                layers_cross = 2, 
                dropout = 0.15,
                head_type=0, 
                ):
        super(EHGNN, self).__init__()
        self.local_model = PaiNN(n_atom_basis = n_atom_basis,
                                n_interactions = layers_local,
                                radial_basis = GaussianRBF(n_rbf = local_cutoff, cutoff = local_cutoff, start = 0.0),
                                cutoff_fn = CosineCutoff(local_cutoff),
                                activation  = F.silu,
                                max_z = 100,
                                epsilon = 1e-8,
                                )
        self.cross_model = BiMP(n_atom_basis = n_atom_basis,
                                n_interactions = layers_cross,
                                radial_basis = GaussianRBF(n_rbf = cross_cutoff, cutoff = cross_cutoff, start = 0.0),
                                cutoff_fn = CosineCutoff(cross_cutoff),
                                activation  = F.silu,
                                max_z = 100,
                                epsilon = 1e-8,
                                )
        self.head_type = head_type
        if self.head_type == 0: # pretrain
            self.head = nn.Sequential(Dense(2*n_atom_basis,2*n_atom_basis,activation=F.relu),nn.Dropout(dropout),Dense(2*n_atom_basis,27,activation=None))
        elif self.head_type == 1: # lba
            self.head = nn.Sequential(Dense(2*n_atom_basis,
                                            2*n_atom_basis,activation=F.relu),
                                      nn.Dropout(dropout),
                                      Dense(2*n_atom_basis,1,activation=None))
        elif self.head_type == 2: # lep
            self.head = nn.Sequential(Dense(2*n_atom_basis,2*n_atom_basis,activation=F.relu),
                                      nn.Dropout(dropout),
                                      Dense(2*n_atom_basis,1,activation=None))
        elif self.head_type == 3: # visialize
            self.head = None
        
    def forward(self,data):
        if self.head_type == 0:
            x, pos, energy,sasa ,x_type,x_res,batch = data.x.long(), data.pos,data.energy,data.sasa,data.x_type,data.x_res,data.batch
        else:
            x, pos, y ,x_type,x_res,batch = data.x.long(), data.pos.float(),data.y,data.x_type,data.x_res,data.batch
        mask = x > 1
        noise = torch.empty(pos[mask].shape, dtype=torch.float32).uniform_(-1,1).cuda()
        x, pos,x_type,x_res,batch = x[mask], pos[mask] + noise,x_type[mask],x_res[mask],batch[mask]
        x_raw = copy.deepcopy(x).cuda()
        mask_atom = None
        if self.head_type == 0:
            # masked atoms
            mask_atom = torch.empty(x.shape, dtype=torch.float32).uniform_(0,1).cuda()
            mask_atom = (mask_atom < 0.15) & (x_type == 0)
            x[mask_atom] = 60
        if self.head_type != 2:
            h,v = self.local_model(x,pos,x_type,None,batch)
        else:
            h,v = self.local_model(x,pos,x_type,x_res,batch)
        
        local_h = h.squeeze(1)
        
        node_features,v = self.cross_model(h,v,pos,x_res,batch)
        
        if self.head is not None:
            node_features = self.head(torch.cat((node_features,local_h),dim=-1))
        outputs = None
        if self.head_type == 0:
            new_atom_type = node_features[:,2:]
            node_features = scatter_mean(node_features, batch, dim=0)
            new_energy = node_features[:,0]
            new_sasa = node_features[:,1]
            outputs = [energy,new_energy,sasa,new_sasa,x_raw[mask_atom],new_atom_type[mask_atom]]
        elif self.head_type != 3:
            new_y = scatter_mean(node_features, batch, dim=0)
            outputs = [y,new_y.squeeze(-1)]
        else:
            node_features = scatter_mean(node_features, batch, dim=0)
            outputs = [y,node_features]
        return outputs