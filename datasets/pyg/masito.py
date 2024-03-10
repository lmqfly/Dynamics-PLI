import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
import numpy as np
import h5py
from tqdm import tqdm
from torch.utils.data import Subset
import atom3d.util.formats as fo
import os

class MASITO(InMemoryDataset):
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
        md_H5File = h5py.File('../../../autodl-tmp/MD.hdf5')
        datalist = []
        keep = []
        with open("sequence/keep_no_more_than_30%_identity.txt","r") as f:
            for line in f.readlines():
                line = line.strip("\n")
                keep.append(line)
        dict = {'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'P': 6, 'S': 7, 'CL': 8, 'BR': 9, 'I': 10, 'NA': 11, 'ZN': 12, 'CA': 13, 'FE': 14, 'MG': 15, 'MN': 16, 'CO': 17, 'CU': 18, 'SR': 19, 'K': 20, 'CS': 21, 'NI': 22, 'CD': 23, 'SI': 24}
              
        res_dict = {'MOL': 0,'ACE': 1,'ALA': 2,'ARG': 3, 'ASN': 4,'ASP': 5,'CYS': 6,'CYX': 7,'GLN': 8,'GLU': 9,'GLY': 10,'HIE': 11, 'ILE': 12,'LEU': 13,'LYS': 14,'MET': 15,'PHE': 16,'PRO': 17,'SER': 18,'THR': 19,'TRP': 20,'TYR': 21,'VAL': 22, 'MG': 0, 'NA': 0, '0': 0, 'CA': 0, 'FE': 0, 'FE2': 0, 'MN': 0, 'CO': 0, 'CU': 0, 'SR': 0, 'K': 0, 'CS': 0, 'NI': 0, 'CD': 0, ' ZN': 0, 'NAG': 0, 'HOH': 0, 'CSO': 0}
        
        for file in tqdm(keep):
            begin_atom_index = md_H5File[file]['molecules_begin_atom_index'][-1]
            protein_positions = md_H5File[file]['trajectory_coordinates'][:,0:begin_atom_index,:]
            ligand_positions = md_H5File[file]['trajectory_coordinates'][:,begin_atom_index:,:]

            # atom3d选的是6A距离
            dist = torch.cdist(torch.tensor(protein_positions[0]), torch.tensor(ligand_positions[0]))
            mask = (torch.sum(dist < 6, dim=-1) > 0)
            mask = torch.cat((mask, torch.ones(ligand_positions.shape[1]))).bool().numpy()

            x = torch.tensor(np.array(md_H5File[file]['atoms_element']))[mask]
            x_type = torch.zeros_like(x)
            x_type[-ligand_positions.shape[1]:] = 1 # 1 for ligand and 0 for pocket
            pos = torch.tensor(np.array(md_H5File[file]['trajectory_coordinates']))[:,mask].permute(1,0,2).float()
            energy = torch.mean(torch.tensor(np.array(md_H5File[file]['frames_interaction_energy'])).float())
            sasa = torch.mean(torch.tensor(np.array(md_H5File[file]['frames_bSASA'])).float())
            x_res = torch.tensor(np.array(md_H5File[file]['atoms_residue']))[mask]
            # print(x.shape,x_type.shape,pos.shape,energy.shape,x_res.shape,sasa.shape)
            if (x>1).sum() >= 50 and (x>1).sum() <= 300:
                for i in range(100):
                    datalist.append(Data(x=x,energy = energy,sasa = sasa,pos=pos[:,i],x_type=x_type,x_res=x_res))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None: 
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(datalist)
        torch.save((data, slices), self.processed_paths[0])