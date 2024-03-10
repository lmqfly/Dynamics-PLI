import torch
from torch import tensor
from torch_geometric.data import InMemoryDataset, download_url, Data
import numpy as np
from atom3d.datasets import LMDBDataset
from torch.utils.data import Subset
import atom3d.util.formats as fo
import os

class LEPDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.root = root
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        dataset = LMDBDataset('../lep/split-by-protein/data/test')
        datalist = []
        dict = {'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'P': 6, 'S': 7, 'CL': 8, 'BR': 9, 'I': 10, 'NA': 11, 'ZN': 12, 'CA': 13, 'FE': 14, 'MG': 15, 'MN': 16, 'CO': 17, 'CU': 18, 'SR': 19, 'K': 20, 'CS': 21, 'NI': 22, 'CD': 23, 'SI': 24}
              
        res_dict = {'MOL': 0,'ACE': 1,'ALA': 2,'ARG': 3, 'ASN': 4,'ASP': 5,'CYS': 6,'CYX': 7,'GLN': 8,'GLU': 9,'GLY': 10,'HIE': 11, 'ILE': 12,'LEU': 13,'LYS': 14,'MET': 15,'PHE': 16,'PRO': 17,'SER': 18,'THR': 19,'TRP': 20,'TYR': 21,'VAL': 22, 'MG': 0, 'NA': 0, '0': 0, 'CA': 0, 'FE': 0, 'FE2': 0, 'MN': 0, 'CO': 0, 'CU': 0, 'SR': 0, 'K': 0, 'CS': 0, 'NI': 0, 'CD': 0, ' ZN': 0, 'NAG': 0, 'HOH': 0, 'CSO': 0}
        
        for i in range(len(dataset)):
            struct = dataset[i]
            if struct['label'] == 'A':
                y = 1
                key = 'atoms_active'
            else:
                y = 0
                key = 'atoms_inactive'

            atoms = struct[key]
            lig = atoms[atoms.resname == 'UNK']
            protein = atoms[atoms.resname != 'UNK']
            dist = torch.cdist(tensor(protein[['x', 'y', 'z']].values), tensor(lig[['x', 'y', 'z']].values))
            mask = (torch.sum(dist <= 6, dim=-1) > 0)
            mask = torch.cat((mask, torch.zeros(len(lig)))).bool().numpy()
            atoms_pocket = atoms[mask]
            atoms_ligand = atoms[atoms.resname == 'UNK']
            
            x_tmp = [] #原子类型
            x_type = [] #属于配体还是靶标
            x_res = [] #氨基酸类型
            for m in atoms_pocket['element']:
                x_type.append(0)
                if m in dict.keys():
                    x_tmp.append(dict[m])
                else:
                    x_tmp.append(len(dict) + 1)
                    dict[m] = len(dict) + 1
                    
            for m in atoms_pocket['resname']:
                if m in res_dict.keys():
                    x_res.append(res_dict[m])
                else:
                    x_res.append(len(res_dict) + 1)
                    res_dict[m] = len(res_dict) + 1
                    
                    
            for m in atoms_ligand['element']:
                x_type.append(1)
                x_res.append(0)
                if m in dict.keys():
                    x_tmp.append(dict[m])
                else:
                    x_tmp.append(len(dict) + 1)
                    dict[m] = len(dict) + 1
             
            x = torch.tensor(x_tmp)
            x_type = torch.tensor(x_type)
            x_res = torch.tensor(x_res)
            pos = torch.tensor(np.append(atoms_pocket[['x', 'y', 'z']].values, atoms_ligand[['x', 'y', 'z']].values, axis=0))
            y=y
            datalist.append(Data(x=x, y=y, pos=pos,x_type=x_type,x_res=x_res))
        print(dict,name_dict,res_dict)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None: 
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(datalist)
        torch.save((data, slices), self.processed_paths[0])