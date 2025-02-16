from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT

import os
import os.path as osp
import pathlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_tar
import pandas as pd

from src import utils
from src.analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges, compute_molecular_metrics
from src.datasets.abstract_dataset import AbstractDatasetInfos, MolecularDataModule


def files_exist(files: Sequence[str]) -> bool:
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


atom_decoder = ['C', 'N', 'O', 'F', 'B', 'Br', 'Cl', 'I', 'P', 'S', 'Se', 'Si']


class ZincDataset(InMemoryDataset):
    raw_url = "http://deepchemdata.s3-us-west-1.amazonaws.com/datasets/zinc15_250K_2D.tar.gz"

    def __init__(self, stage, root, filter_dataset: bool=False, transform=None, pre_transform=None, pre_filter=None):
        self.stage = stage
        self.atom_decoder = atom_decoder
        self.filter_dataset = filter_dataset
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        # Expected raw files in raw_dir
        return ["zinc15_250K_2D.csv"]

    @property
    def split_file_name(self):
        # name for each data file
        return ["train_zinc.csv", "val_zinc.csv", "test_zinc.csv"]

    @property
    def split_paths(self):
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        if self.filter_dataset:
            return ['train_filtered.pt', 'val_filtered.pt', 'test_filtered.pt']
        else:
            return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        try:
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_tar(file_path, self.raw_dir, mode='r:gz')
            os.unlink(file_path)
        except Exception as e:
            raise RuntimeError(f"Error downloading ZINC dataset: {e}")
        
        dataset = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names[0]))

        n_train = 25000
        n_val = 2000
        n_test = 2000

        train, val, test, _ = np.split(dataset.sample(frac=1, random_state=0), [n_train, n_val + n_train, n_val + n_train + n_test])
        train.to_csv(os.path.join(self.raw_dir, 'train_zinc.csv'))
        val.to_csv(os.path.join(self.raw_dir, 'val_zinc.csv'))
        test.to_csv(os.path.join(self.raw_dir, 'test_zinc.csv'))


    def process(self):
        RDLogger.DisableLog('rdApp.*')

        types = {atom: i for i, atom in enumerate(self.atom_decoder)}

        bonds = {
            BT.SINGLE: 0,
            BT.DOUBLE: 1,
            BT.TRIPLE: 2,
            BT.AROMATIC: 3
        }

        path = self.split_paths[self.file_idx]
        smiles_list = pd.read_csv(path)['smiles'].tolist()

        data_list = []
        smiles_kept = []

        for i, smile in enumerate(tqdm(smiles_list, desc=f"Processing ZINC {self.stage} split")):
            mol = Chem.MolFromSmiles(smile)
            N = mol.GetNumAtoms()
            type_idx = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                b_type = bonds[bond.GetBondType()]
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()] + 1]
            if len(row) == 0:
                continue
            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds)+1).to(torch.float)
            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]
            x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
            y = torch.zeros((1, 0), dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)
            if self.filter_dataset:
                dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
                dense_data = dense_data.mask(node_mask, collapse=True)
                X, E = dense_data.X, dense_data.E
                assert X.size(0) == 1
                atom_types = X[0]
                edge_types = E[0]
                mol = build_molecule_with_partial_charges(atom_types, edge_types, atom_decoder)
                smiles = mol2smiles(mol)
                if smiles is not None:
                    try:
                        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                        if len(mol_frags) == 1:
                            data_list.append(data)
                            smiles_kept.append(smiles)
                    except Chem.rdchem.AtomValenceException:
                        print("Valence error in GetmolFrags")
                    except Chem.rdchem.KekulizeException:
                        print("Can't kekulize molecule")
            else:
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
        
        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

        # Save the smiles strings for the train split
        if self.filter_dataset:
            smiles_save_path = osp.join(pathlib.Path(self.raw_paths[0]).parent, f'{self.stage}.smiles')
            np.save(smiles_save_path, np.array(smiles_kept))
            print(f"Number of molecules kept: {len(smiles_kept)} / {len(smiles_list)}")
        else:
            smiles_save_path = osp.join(pathlib.Path(self.raw_paths[0]).parent, f'{self.stage}.smiles')
            np.save(smiles_save_path, np.array(smiles_list))


class ZincDataModule(MolecularDataModule):
    """
    A MolecularDataModule for the ZINC dataset.
    
    This class creates train/val/test splits using ZincDataset and integrates with
    your overall pipeline.
    """
    def __init__(self, cfg):
        self.remove_h = False
        self.datadir = cfg.dataset.datadir
        self.filter_dataset = cfg.dataset.filter
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = osp.join(base_path, self.datadir)
        datasets = {
            'train': ZincDataset(stage='train', root=root_path),
            'val': ZincDataset(stage='val', root=root_path),
            'test': ZincDataset(stage='test', root=root_path)
        }
        super().__init__(cfg, datasets)


class Zincinfos(AbstractDatasetInfos):
    def __init__(self, datamodule: MolecularDataModule, recompute_statistics=False, meta=None):
        self.name = 'ZINC'
        self.input_dims = None
        self.output_dims = None
        self.remove_h = False
        # Use the same atom mapping as in ZincDataset.
        self.atom_decoder = atom_decoder
        self.atom_encoder = {atom: i for i, atom in enumerate(self.atom_decoder)}
        self.num_atom_types = len(self.atom_encoder)
        self.atom_weights = {
            0: 12, 1: 14, 2: 16, 3: 19, 4: 10.81, 5: 79.9,
            6: 35.45, 7: 126.9, 8: 30.97, 9: 30.07, 10: 78.97, 11: 28.09
        }
        self.max_weight = 349.91
        self.valencies = [4, 3, 2, 1, 3, 1, 1, 1, 3, 2, 2, 4]
        self.n_nodes = torch.tensor(
            [
                0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                0.0000e+00, 5.8824e-05, 1.7647e-04, 7.6471e-04, 3.0588e-03, 1.2882e-02,
                4.5118e-02, 7.9765e-02, 1.0871e-01, 1.3894e-01, 1.6088e-01, 1.8771e-01,
                1.4718e-01, 1.1165e-01, 3.1176e-03
            ]
        )
        self.node_types = torch.tensor(
            [
                6.9992e-01, 1.5766e-01, 1.1976e-01, 1.0817e-02, 3.0361e-06, 3.9772e-04,
                2.5533e-03, 6.0721e-06, 1.5180e-05, 8.8562e-03, 0.0000e+00, 3.0361e-06
            ]
        )
        self.edge_types = torch.tensor([8.9952e-01, 6.6496e-02, 7.7541e-03, 2.9075e-04, 2.5936e-02])
        self.max_n_nodes = len(self.n_nodes) - 1
        self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
        self.valency_distribution[:7] = torch.tensor([0.0000, 0.1191, 0.3916, 0.2949, 0.1892, 0.0018, 0.0034])

        if recompute_statistics:
            self.n_nodes = datamodule.node_counts()
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt('zinc_n_counts.txt', self.n_nodes.numpy())
            self.node_types = datamodule.node_types()
            print("Distribution of node types", self.node_types)
            np.savetxt('zinc_atom_types.txt', self.node_types.numpy())
            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt('zinc_edge_types.txt', self.edge_types.numpy())
            self.max_n_nodes = len(self.n_nodes) - 1
            valencies = datamodule.valency_count(self.max_n_nodes)
            print("Distribution of valencies", valencies)
            np.savetxt('zinc_valencies.txt', valencies.numpy())
            self.valency_distribution = valencies        
        super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)


if __name__ == "__main__":
    ZincDataset("val", os.path.join(os.path.abspath(__file__), "../../../data/zinc"))
