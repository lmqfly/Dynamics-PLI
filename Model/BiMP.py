from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph
from .utils import Blocking,covert_res_type_to_index,Dense,VectorNorm
from torch_geometric.nn import radius
from torch_scatter import scatter_add
class BI_Interaction(nn.Module):
    r"""PaiNN interaction block for modeling equivariant interactions of atomistic systems."""

    def __init__(self, n_atom_basis: int, activation: Callable):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            activation: if None, no activation function is used.
            epsilon: stability constant added in norm to prevent numerical instabilities
        """
        super(BI_Interaction, self).__init__()
        self.n_atom_basis = n_atom_basis

        self.interatomic_context_net = nn.Sequential(
            Dense(n_atom_basis, n_atom_basis, activation=activation),
            Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )
        
        self.v_channel_mix = Dense(
            n_atom_basis, 2 * n_atom_basis, activation=None, bias=False
        )
        
        self.h_remain = Dense(
            n_atom_basis, n_atom_basis, activation=None, bias=False
        )
        
        self.h_forget = Dense(
            n_atom_basis, n_atom_basis, activation=None, bias=False
        )
        
        self.v_remain = Dense(
            n_atom_basis, 1, activation=None, bias=False
        )
        
        self.layernorm = nn.LayerNorm(n_atom_basis)
        
    def forward(
        self,
        h: torch.Tensor,
        v: torch.Tensor,
        H: torch.Tensor,
        V: torch.Tensor,
        Wij: torch.Tensor,
        dir_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        n_atoms: int,
    ):
        """Compute interaction output.

        Args:
            q: scalar input values
            mu: vector input values
            Wij: filter
            idx_i: index of center atom i
            idx_j: index of neighbors j

        Returns:
            atom features after interaction
        """
        # inter-atomic
        X = self.interatomic_context_net(H)
        Hj = X[idx_j]
        Vj = V[idx_j]
        X = Wij * Hj
        dh, dvR, dvv = torch.split(X, self.n_atom_basis, dim=-1)
        dh = scatter_add(dh, idx_i, dim=0,dim_size=n_atoms)
        dv = dvR * dir_ij[..., None] + dvv * Vj
        dv = scatter_add(dv, idx_i, dim=0,dim_size=n_atoms)
        
        v_mix = self.v_channel_mix(v)
        v_V, v_W = torch.split(v_mix, self.n_atom_basis, dim=-1)
        q = self.layernorm(h + self.h_remain(dh) + self.h_forget(dh) * torch.sum(v_V * v_W, dim=1, keepdim=True))
        mu = VectorNorm(v + dv + self.v_remain(dh) * v)
        
        return q, mu

class BiMP(nn.Module):

    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        radial_basis: nn.Module,
        cutoff_fn: Optional[Callable] = None,
        activation: Optional[Callable] = F.silu,
        max_z: int = 100,
        shared_interactions: bool = False,
        shared_filters: bool = False,
        epsilon: float = 1e-8,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks.
            radial_basis: layer for expanding interatomic distances in a basis set
            cutoff_fn: cutoff function
            activation: activation function
            shared_interactions: if True, share the weights across
                interaction blocks.
            shared_interactions: if True, share the weights across
                filter-generating networks.
            epsilon: stability constant added in norm to prevent numerical instabilities
        """
        super(BiMP, self).__init__()

        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.radial_basis = radial_basis
    
        self.residue_EMB = nn.Embedding(100,n_atom_basis)
        self.share_filters = shared_filters

        if shared_filters:
            self.filter_net = Dense(
                self.radial_basis.n_rbf, 3 * n_atom_basis, activation=None
            )
        else:
            self.filter_net = Dense(
                self.radial_basis.n_rbf,
                self.n_interactions * n_atom_basis * 3,
                activation=None,
            )

        self.interactions = nn.ModuleList([
            BI_Interaction(
                n_atom_basis=self.n_atom_basis, activation=activation
            ) 
            for i in range(self.n_interactions)])

    def forward(self, h,v,pos,x_res,batch):
        """
        Compute atomic representations/embeddings.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.
        """
        res_index = covert_res_type_to_index(x_res) #blocking indice
    
        H, V, POS, RES, batch_block = Blocking(h, v, pos, x_res, res_index, batch)
        H = H + self.residue_EMB(RES).unsqueeze(1)
        idx_i, idx_j = radius(POS,pos,self.cutoff,batch_block,batch)
        
        r_ij = pos[idx_i] - POS[idx_j]
        n_atoms = h.shape[0]
            
        # compute atom and pair features
        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / (d_ij + 1e-6)
        phi_ij = self.radial_basis(d_ij)
        fcut = self.cutoff_fn(d_ij)

        filters = self.filter_net(phi_ij) * fcut[..., None]

        if self.share_filters:
            filter_list = [filters] * self.n_interactions
        else:
            filter_list = torch.split(filters, 3 * self.n_atom_basis, dim=-1)
        for i, interaction in enumerate(self.interactions):
            if i!= 0 : # the complex has been blocked for the first layer
                H, V, POS, RES, batch_block = Blocking(h, v, pos, x_res, res_index, batch)
            
            h, v = interaction(h, v, H, V, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            
        h = h.squeeze(1)

        return h,v
