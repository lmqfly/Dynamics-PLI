import torch
from torch import nn, einsum, broadcast_tensors
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .utils import KNNBatchEdgeConstructor
from torch_geometric.nn import radius_graph
from torch_scatter import scatter_mean,scatter_add
# types
import torch.nn.functional as F
from typing import Optional, List, Union
# pytorch geometric

try:
    import torch_geometric
    from torch_geometric.nn import MessagePassing
    from torch_geometric.typing import Adj, Size, OptTensor, Tensor
except:
    Tensor = OptTensor = Adj = MessagePassing = Size = object
    PYG_AVAILABLE = False

    # to stop throwing errors from type suggestions
    Adj = object
    Size = object
    OptTensor = object
    Tensor = object

import torch
from torch import nn


class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False, normalize=False, tanh=False):
        '''

        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.ele_embedding = nn.Embedding(100,in_node_nf)
        self.res_embedding = nn.Embedding(100,in_node_nf)
        self.pos_embedding = nn.Embedding(100,in_node_nf)
        self.edge_embedding = nn.Embedding(4,in_edge_nf)
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh))
        self.edge_constructor = KNNBatchEdgeConstructor(
            k_neighbors=9,
            delete_self_loop=True,
            global_node_id_vocab=[98]
            )
        
        self.head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 1, bias=False)
        )

    def forward(self, data):
        z,pos,y,z_type,z_res,batch = data.x.long(), data.pos.float(),data.y,data.x_type,data.x_res,data.batch
        mask = torch.logical_or(z > 1,z_type == 1)
        noise = torch.empty(pos[mask].shape, dtype=torch.float32).uniform_(-1,1).cuda()
        # noise = torch.zeros(pos[mask].shape,dtype=torch.float32).cuda()
        z,pos,z_type,z_res,batch = z[mask], pos[mask] + noise,z_type[mask],z_res[mask],batch[mask]

        intra_edges, inter_edges, global_normal_edges, global_global_edges = self.edge_constructor(z, batch, z_type,X=pos)
        edge_index = torch.cat([intra_edges, inter_edges, global_global_edges, global_normal_edges], dim=1)
        edge_attr = torch.cat([
            torch.zeros_like(intra_edges[0]),
            torch.ones_like(inter_edges[0]),
            torch.ones_like(global_global_edges[0]) * 2,
            torch.ones_like(global_normal_edges[0]) * 3])
        
        # h = self.ele_embedding(z) + self.res_embedding(z_res) + self.res_embedding(z_pos)
        h = self.ele_embedding(z) + self.res_embedding(z_res)
        h = self.embedding_in(h)
        edge_attr = self.edge_embedding(edge_attr)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edge_index, pos, edge_attr)
        h = self.embedding_out(h)
        h = F.normalize(h,dim=-1)
        h = self.head(h)
        new_y= scatter_add(h, batch, dim=0)
        outputs = [y, new_y.squeeze(-1)]
        return outputs
    

def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

