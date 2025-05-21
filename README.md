# Dynamics-PLI

Official PyTorch-based implementation of the paper:  
**"Molecular dynamics-powered hierarchical geometric deep learning framework for protein-ligand interaction"**  
[[IEEE Xplore]](https://ieeexplore.ieee.org/document/10955744) | [[DOI]](https://doi.org/10.1093/bib/bbad404)

In this work, we introduce **Dynamics-PLI**, a **SO(3)-equivariant hierarchical graph neural network (EHGNN)** designed to capture the intrinsic hierarchy of biomolecular structures. This framework leverages **molecular dynamics simulations** to enhance the prediction of protein-ligand interactions (PLIs), providing a powerful tool for computational drug discovery.

## 📦 Installation
```
git clone https://github.com/yourusername/Dynamics-PLI.git
cd Dynamics-PLI
conda env create -f environment.yml
conda activate Dynamics-PLI
```

## 🔮 Dataprocess
### download

The following datasets and tools were utilized in the development and evaluation of our model:

- **MISATO**  
  A multi-scale integrative tool for protein structure analysis, particularly useful for studying conformational changes and interactions.  
  [🔗 Zenodo Record](https://zenodo.org/records/7711953)

- **Atom3D**  
  A collection of benchmark datasets for machine learning on 3D molecular structures, designed to facilitate research in structural biology.  
  [🔗 Official Website](https://www.atom3d.ai/)


### MD_data_process
```
python ./datasets/pyg/MISATO.py
```
### downstream_data_process
```
python ./datasets/pyg/pdbbind.py
python ./datasets/pyg/lep.py
```
## 🚀 Quick Start
```
python lba_main_count.py --dim 128 --dropout 0.15  --cross_cutoff 20 --local_cutoff 10 --depth_local 6 --depth_cross 2  --seed 4 --epochs 10 --atoms_bacth 4000 --lr 0.00005
```
📁 Project Structure
```
Dynamics-PLI/
├── baseline/            # baseline methods
├── datasets/            # Dataset loaders and preprocessors
├── Model/              # Model
├── util_mine.py/        # Training utilities, metrics, logging
├── lba_main_count.py    # lba task scripts
├── lep_main_count.py    # lep task scripts
├── pretrain.py          # pretrain scripts
└── README.md            # Project documentation

```
### 📊 Results


↑: Higher is better, ↓: Lower is better.

| Pre-trained Model        | LBA (30%) RMSE ↓      | Pearson ↑           | Spearman ↑          | LEP AUROC ↑         | AUPRC ↑            |
|--------------------------|------------------------|----------------------|----------------------|----------------------|---------------------|
| EGNN + Pre [[Satorras et al., 2021]] | _1.341 ± 0.027_        | _0.617 ± 0.016_      | _0.613 ± 0.013_      | _0.759 ± 0.010_      | _0.752 ± 0.023_     |
| **Dynamics-PLI**         | **1.312 ± 0.005**       | **0.636 ± 0.005**    | **0.633 ± 0.008**    | **0.779 ± 0.007**    | **0.766 ± 0.031**   |
| **Δ (%)**                | ↓ 4.0%                 | ↑ 3.5%               | ↑ 4.6%               | ↑ 3.7%               | ↑ 4.2%              |



Full benchmarking details are available in the paper.

## 📚 Reference
If you use Dynamics-PLI in scholarly publications, presentations, or to communicate with your satellite, please cite the following work:


```
@ARTICLE{10955744,
  author={Liu, Mingquan and Jin, Shuting and Lai, Houtim and Wang, Longyue and Wang, Jianmin and Cheng, Zhixiang and Zeng, Xiangxiang},
  journal={IEEE Transactions on Computational Biology and Bioinformatics}, 
  title={Molecular dynamics-powered hierarchical geometric deep learning framework for protein-ligand interaction}, 
  year={2025},
  volume={},
  number={},
  pages={1-12},
  keywords={Proteins;Atoms;Deep learning;Three-dimensional displays;Graph neural networks;Feature extraction;Drugs;Computational modeling;Training;Representation learning;Protein-ligand interactions;Molecular dynamics;Geometric deep learning;Pre-training},
  doi={10.1109/TCBBIO.2025.3558959}
}
```

## 🤝 Acknowledgements
### Acknowledgements

We gratefully acknowledge the following works, which provided essential datasets and tools for this research:

- **MISATO: Machine Learning Dataset of Protein–Ligand Complexes for Structure-Based Drug Discovery**  
  This work offers a valuable dataset for training and evaluating models in protein–ligand interaction prediction.  
  [🔗 GitHub Repository](https://github.com/t7morgen/misato-dataset)

- **ATOM3D: Tasks On Molecules in 3 Dimensions**  
  This project provides a suite of benchmark tasks for machine learning on 3D molecular structures, advancing research in structural biology and drug discovery.  
  [🔗 GitHub Repository](https://github.com/drorlab/atom3d)
