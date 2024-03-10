import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
import numpy as np
from atom3d.datasets import LMDBDataset
from torch.utils.data import Subset
import atom3d.util.formats as fo
import os

class PdbbindDataset_glb(InMemoryDataset): # with global node
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.root = root
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        # pass # 不能使用pass，会报join() argument must be str or bytes, not 'NoneType'错误
        return []

    # 首先寻找processed_paths[0]路径下的文件名也就是之前process方法保存的文件名
    @property
    def processed_file_names(self):
        return ['data.pt']

    # 用于从网上下载数据集，下载原始数据到指定的文件夹下，自己的数据集可以跳过
    def download(self):
        pass

    # 生成数据集所用的方法，程序第一次运行才执行并生成processed文件夹的处理过后数据的文件，否则必须删除已经生成的processed文件夹中的所有文件才会重新执行此函数
    def process(self):
        dataset = LMDBDataset(self.root[:-2])
        datalist = []
        dict = {'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'P': 6, 'S': 7, 'CL': 8, 'BR': 9, 'I': 10, 'NA': 11, 'ZN': 12, 'CA': 13, 'FE': 14, 'MG': 15, 'MN': 16, 'CO': 17, 'CU': 18, 'SR': 19, 'K': 20, 'CS': 21, 'NI': 22, 'CD': 23, 'SI': 24,'GLB':98}
              
        res_dict = {'MOL': 0,'ACE': 1,'ALA': 2,'ARG': 3, 'ASN': 4,'ASP': 5,'CYS': 6,'CYX': 7,'GLN': 8,'GLU': 9,'GLY': 10,'HIE': 11, 'ILE': 12,'LEU': 13,'LYS': 14,'MET': 15,'PHE': 16,'PRO': 17,'SER': 18,'THR': 19,'TRP': 20,'TYR': 21,'VAL': 22, 'MG': 0, 'NA': 0, '0': 0, 'CA': 0, 'FE': 0, 'FE2': 0, 'MN': 0, 'CO': 0, 'CU': 0, 'SR': 0, 'K': 0, 'CS': 0, 'NI': 0, 'CD': 0, ' ZN': 0, 'NAG': 0, 'HOH': 0, 'CSO': 0,'GLB':98}
                    
        pos_dict = {'': 1, 'N': 2, 'A': 3, 'B': 4, 'G': 5, 'D': 6, 'E': 7, 'Z': 8, 'H': 9, 'XT': 10, 'MOL': 0,'GLB':98}
        
        for i in range(len(dataset)):
            struct = dataset[i]
            atoms_pocket = struct['atoms_pocket']
            atoms_ligand = struct['atoms_ligand']
            x_tmp = []
            x_type = []
            x_res = []
            x_pos = []
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
                    
            for m,e in zip(atoms_pocket['name'],atoms_pocket['element']):
                m = ''.join((c for c in m if not c.isdigit())) 
                m = m.lstrip(e)
                if m in pos_dict.keys():
                    x_pos.append(pos_dict[m])
                else:
                    x_pos.append(len(pos_dict) + 1)
                    pos_dict[m] = len(pos_dict) + 1
                    
            for m in atoms_ligand['element']:
                x_type.append(1)
                x_res.append(0)
                x_pos.append(0)
                if m in dict.keys():
                    x_tmp.append(dict[m])
                else:
                    x_tmp.append(len(dict) + 1)
                    dict[m] = len(dict) + 1
         
            pos_1, pos_2 = fo.get_coordinates_from_df(atoms_pocket), fo.get_coordinates_from_df(atoms_ligand)
            # global_node_pocket
            x_tmp.append(98)
            x_type.append(0)
            x_res.append(98)
            x_pos.append(98)
            
            # global_node_ligand
            x_tmp.append(98)
            x_type.append(1)
            x_res.append(98)
            x_pos.append(98)
            
            x = torch.tensor(x_tmp)
            x_type = torch.tensor(x_type)
            x_res = torch.tensor(x_res)
            x_pos = torch.tensor(x_pos)
            pos = torch.tensor(np.concatenate((pos_1, pos_2,np.mean(pos_1,axis=0,keepdims=True),np.mean(pos_2,axis=0,keepdims=True)), axis=0))
            y=struct['scores']['neglog_aff']
            datalist.append(Data(x=x, y=y, pos=pos,x_type=x_type,x_res=x_res,x_pos=x_pos))
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None: 
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(datalist)
        torch.save((data, slices), self.processed_paths[0])
        
class PdbbindDataset(InMemoryDataset): # without global node
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.root = root
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        # pass # 不能使用pass，会报join() argument must be str or bytes, not 'NoneType'错误
        return []

    # 首先寻找processed_paths[0]路径下的文件名也就是之前process方法保存的文件名
    @property
    def processed_file_names(self):
        return ['data.pt']

    # 用于从网上下载数据集，下载原始数据到指定的文件夹下，自己的数据集可以跳过
    def download(self):
        pass

    # 生成数据集所用的方法，程序第一次运行才执行并生成processed文件夹的处理过后数据的文件，否则必须删除已经生成的processed文件夹中的所有文件才会重新执行此函数
    def process(self):
        dataset = LMDBDataset('../pdbbind/train')
        datalist = []
        dict = {'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'P': 6, 'S': 7, 'CL': 8, 'BR': 9, 'I': 10, 'NA': 11, 'ZN': 12, 'CA': 13, 'FE': 14, 'MG': 15, 'MN': 16, 'CO': 17, 'CU': 18, 'SR': 19, 'K': 20, 'CS': 21, 'NI': 22, 'CD': 23, 'SI': 24}
              
        res_dict = {'MOL': 0,'ACE': 1,'ALA': 2,'ARG': 3, 'ASN': 4,'ASP': 5,'CYS': 6,'CYX': 7,'GLN': 8,'GLU': 9,'GLY': 10,'HIE': 11, 'ILE': 12,'LEU': 13,'LYS': 14,'MET': 15,'PHE': 16,'PRO': 17,'SER': 18,'THR': 19,'TRP': 20,'TYR': 21,'VAL': 22, 'MG': 0, 'NA': 0, '0': 0, 'CA': 0, 'FE': 0, 'FE2': 0, 'MN': 0, 'CO': 0, 'CU': 0, 'SR': 0, 'K': 0, 'CS': 0, 'NI': 0, 'CD': 0, ' ZN': 0, 'NAG': 0, 'HOH': 0, 'CSO': 0}
        
        for i in range(len(dataset)):
            struct = dataset[i]
            atoms_pocket = struct['atoms_pocket']
            atoms_ligand = struct['atoms_ligand']
            x_tmp = []
            x_type = []
            x_res = []
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
         
            pos_1, pos_2 = fo.get_coordinates_from_df(atoms_pocket), fo.get_coordinates_from_df(atoms_ligand)
            x = torch.tensor(x_tmp)
            x_type = torch.tensor(x_type)
            x_res = torch.tensor(x_res)
            pos = torch.tensor(np.append(pos_1, pos_2, axis=0))
            y=struct['scores']['neglog_aff']
            print(x.shape,x_type.shape,x_res.shape,pos.shape)
            datalist.append(Data(x=x, y=y, pos=pos,x_type=x_type,x_res=x_res))
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None: 
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(datalist)
        torch.save((data, slices), self.processed_paths[0])