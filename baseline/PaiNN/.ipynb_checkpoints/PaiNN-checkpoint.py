from typing import Callable, Dict, Optional
from .utils import GaussianRBF,CosineCutoff
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import KNNBatchEdgeConstructor
from torch_geometric.nn import radius_graph
from .utils import Dense,VectorNorm
from torch_scatter import scatter_add    
from torch_scatter import scatter_mean

class PaiNNInteraction(nn.Module):
    r"""PaiNN interaction block for modeling equivariant interactions of atomistic systems."""

    def __init__(self, n_atom_basis: int, activation: Callable):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            activation: if None, no activation function is used.
            epsilon: stability constant added in norm to prevent numerical instabilities
        """
        super(PaiNNInteraction, self).__init__()
        self.n_atom_basis = n_atom_basis

        self.interatomic_context_net = nn.Sequential(
            Dense(n_atom_basis, n_atom_basis, activation=activation),
            Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )

    def forward(
        self,
        q: torch.Tensor,
        mu: torch.Tensor,
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
        # print(7,torch.any(torch.isnan(q)))
        x = self.interatomic_context_net(q)
        # print(8,torch.any(torch.isnan(x)))
        xj = x[idx_j]
        muj = mu[idx_j]
        x = Wij * xj
        # print(9,torch.any(torch.isnan(Wij)),torch.any(torch.isnan(dir_ij)),torch.any(torch.isnan(x)))
        dq, dmuR, dmumu = torch.split(x, self.n_atom_basis, dim=-1)
        # print(10,torch.any(torch.isnan(dq)))
        dq = scatter_add(dq, idx_i, dim=0,dim_size=n_atoms)
        # print(11,torch.any(torch.isnan(dq)))
        dmu = dmuR * dir_ij[..., None] + dmumu * muj
        dmu = scatter_add(dmu, idx_i, dim=0,dim_size=n_atoms)
        # print(9,torch.any(torch.isnan(dmu)))
        q = q + dq
        mu = mu + dmu
        
        return q, mu


class PaiNNMixing(nn.Module):
    r"""PaiNN interaction block for mixing on atom features."""

    def __init__(self, n_atom_basis: int, activation: Callable, epsilon: float = 1e-8):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            activation: if None, no activation function is used.
            epsilon: stability constant added in norm to prevent numerical instabilities
        """
        super(PaiNNMixing, self).__init__()
        self.n_atom_basis = n_atom_basis

        self.intraatomic_context_net = nn.Sequential(
            Dense(2 * n_atom_basis, n_atom_basis, activation=activation),
            Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )
        self.mu_channel_mix = Dense(
            n_atom_basis, 2 * n_atom_basis, activation=None, bias=False
        )
        self.epsilon = epsilon
        
        self.layernorm = nn.LayerNorm(n_atom_basis)

    def forward(self, q: torch.Tensor, mu: torch.Tensor):
        """Compute intraatomic mixing.

        Args:
            q: scalar input values
            mu: vector input values

        Returns:
            atom features after interaction
        """
        ## intra-atomic
        mu_mix = self.mu_channel_mix(mu)
        mu_V, mu_W = torch.split(mu_mix, self.n_atom_basis, dim=-1)
        mu_Vn = torch.sqrt(torch.sum(mu_V**2, dim=-2, keepdim=True) + self.epsilon)
        # print(6,torch.any(torch.isnan(mu_Vn)))
        ctx = torch.cat([q, mu_Vn], dim=-1)
        x = self.intraatomic_context_net(ctx)

        dq_intra, dmu_intra, dqmu_intra = torch.split(x, self.n_atom_basis, dim=-1)
        dmu_intra = dmu_intra * mu_W

        dqmu_intra = dqmu_intra * torch.sum(mu_V * mu_W, dim=1, keepdim=True)
        # print(7,torch.any(torch.isnan(dqmu_intra)))
        q = self.layernorm(q + dq_intra + dqmu_intra)
        mu = VectorNorm(mu + dmu_intra)
        return q, mu


class PaiNN(nn.Module):
    """PaiNN - polarizable interaction neural network

    References:

    .. [#painn1] Schütt, Unke, Gastegger:
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html

    """

    def __init__(
        self,
        n_atom_basis: int = 128,
        n_interactions: int = 6,
        radial_basis: nn.Module = GaussianRBF(n_rbf = 10, cutoff = 10, start = 0.0),
        cutoff_fn: Optional[Callable] = CosineCutoff(10),
        activation: Optional[Callable] = F.silu,
        max_z: int = 100,
        shared_interactions: bool = False,
        shared_filters: bool = False,
        epsilon: float = 1e-8,
        dropout: float = 0.1,
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
        super(PaiNN, self).__init__()

        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.radial_basis = radial_basis
        self.dropout = dropout
        self.edge_embedding = nn.Embedding(4, self.radial_basis.n_rbf)
        # layers
        self.ele_embedding = nn.Embedding(100,self.n_atom_basis)
        self.res_embedding = nn.Embedding(100,self.n_atom_basis)
        self.pos_embedding = nn.Embedding(100,self.n_atom_basis)
        self.edge_constructor = KNNBatchEdgeConstructor(
            k_neighbors=9,
            delete_self_loop=True,
            global_node_id_vocab=[98]
            )
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
            PaiNNInteraction(
                n_atom_basis=self.n_atom_basis, activation=activation
            ) 
            for i in range(self.n_interactions)])

        self.mixing = nn.ModuleList([
            PaiNNMixing(
                n_atom_basis=self.n_atom_basis, activation=activation, epsilon=epsilon
            )
            for i in range(self.n_interactions)])
        self.head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_atom_basis, n_atom_basis),
            nn.SiLU(),
            nn.Linear(n_atom_basis, 1, bias=False)
        )

    def forward(self, data):
        """
        Compute atomic representations/embeddings.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.
        """
        z,pos,y,z_type,z_res,batch = data.x.long(), data.pos.float(),data.y,data.x_type,data.x_res,data.batch
        mask = torch.logical_or(z > 1,z_type == 1)
        # noise = torch.zeros(pos[mask].shape,dtype=torch.float32).cuda()
        noise = torch.empty(pos[mask].shape, dtype=torch.float32).uniform_(-1,1).cuda()
        z,pos,z_type,z_res,batch = z[mask], pos[mask] + noise,z_type[mask],z_res[mask],batch[mask]
        
        intra_edges, inter_edges, global_normal_edges, global_global_edges = self.edge_constructor(z, batch, z_type,X=pos)
        edge_index = torch.cat([intra_edges, inter_edges, global_global_edges, global_normal_edges], dim=1)
        edge_attr = torch.cat([
            torch.zeros_like(intra_edges[0]),
            torch.ones_like(inter_edges[0]),
            torch.ones_like(global_global_edges[0]) * 2,
            torch.ones_like(global_normal_edges[0]) * 3])
        
        idx_i, idx_j = edge_index
        n_atoms = z.shape[0]
        # compute atom and pair features
        r_ij = pos[idx_i] - pos[idx_j]
        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / (d_ij + 1e-6)
        phi_ij = self.radial_basis(d_ij) + self.edge_embedding(edge_attr).unsqueeze(1)
        fcut = self.cutoff_fn(d_ij)
        
        filters = self.filter_net(phi_ij)
        if self.share_filters:
            filter_list = [filters] * self.n_interactions
        else:
            filter_list = torch.split(filters, 3 * self.n_atom_basis, dim=-1)

        q = self.ele_embedding(z)[:, None] + self.res_embedding(z_res)[:, None]
        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)
        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
            q, mu = interaction(q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)
        h = F.normalize(q.squeeze(1),dim=-1)
        h = self.head(h)
        new_y= scatter_add(h, batch, dim=0)
        outputs = [y, new_y.squeeze(-1)]
        return outputs
