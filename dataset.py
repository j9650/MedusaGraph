import os
import ast
import glob
import torch
import numpy as np
from scipy.sparse import csr_matrix

from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data

import time

# https://github.com/rusty1s/pytorch_geometric/blob/462df68c04ae1e6782b5591928c20f3d506207a3/torch_geometric/datasets/zinc.py
# https://github.com/rusty1s/pytorch_geometric/blob/68ca38e28c1b86cf267fab8499c5f3e52d8ec410/torch_geometric/datasets/ppi.py


class PDBBind(InMemoryDataset):

    def __init__(self, root, subset=False, split='train', transform=None,
                 pre_transform=None, pre_filter=None):
        self.subset = subset
        # super(PDBBind, self).__init__(root, transform, pre_transform, pre_filter)
        super().__init__(root, transform, pre_transform, pre_filter)
        path = os.path.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [
            'train', 'test'
        ]

    @property
    def processed_dir(self):
        # name = 'subset' if self.subset else 'full'
        # return os.path.join(self.root, name, 'processed')
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return ['train.pt', 'test.pt']

    # TODO: implement after uploading the entire dataset somewhere
    def download(self):
        print('Hello Download')
        pass

    def process(self):
        for split in ['train', 'test']:

            dataset_dir = os.path.join(self.raw_dir, f'{split}')
            files_num = len(
                glob.glob(os.path.join(dataset_dir, '*_data-G.json')))

            data_list = []
            graph_idx = 0

            pbar = tqdm(total=files_num)
            pbar.set_description(f'Processing {split} dataset')
        

            for f in range(files_num):

                with open(os.path.join(dataset_dir, f'{f}_data-G.json')) as gf:
                    graphs = gf.readlines()
                num_graphs_per_file = len(graphs)//3

                pbar.total = num_graphs_per_file * files_num
                pbar.refresh()

                # features = []
                # with open(os.path.join(dataset_dir, f'{f}_data-feats'), 'rb') as ff:
                #     for _ in range(num_graphs_per_file):
                #         features.append(np.load(ff))

                feat_file = open(os.path.join(dataset_dir, f'{f}_data-feats'), 'rb')

                labels = np.loadtxt(os.path.join(dataset_dir, f'{f}_label'))
                rmsds = labels[:, 1]
                energies = labels[:, 2]
                labels = labels[:, 0]

                for idx in range(num_graphs_per_file):

                    # it's slow because of these!
                    features = np.load(feat_file)
                    indptr = ast.literal_eval(graphs[3*idx])
                    indices = ast.literal_eval(graphs[3*idx+1])
                    dist = ast.literal_eval(graphs[3*idx+2])
                    
                    # using scipy, probably slow
                    # value = np.ones_like(indices)
                    # # add 0 to indptr for scipy csr
                    # num_nodes = len(indptr)
                    # indptr.insert(0, 0)
                    # coo_graph = csr_matrix((value, indices, indptr), shape=(
                    #     num_nodes, num_nodes)).tocoo()
                    # per graph data
                    # edge_index = torch.Tensor(
                    #     [coo_graph.row, coo_graph.col]).to(torch.long)


                    indptr = torch.LongTensor(indptr)
                    indices = torch.LongTensor(indices)
                    dist = torch.tensor(dist, dtype=torch.float)
                    row_idx = torch.ops.torch_sparse.ind2ptr(indptr,len(indices))[1:]
                    edge_index = torch.stack((row_idx, indices), dim=0)
                    
                    # No edge attr to save space
                    # edge_attr = torch.Tensor(coo_graph.data).to(torch.long)
                    
                    # x = torch.Tensor(features[idx])
                    x = torch.Tensor(features)
                    y = torch.LongTensor([labels[idx].astype(int)])
                    dist = dist.reshape(dist.size()[0], 1)
                    # e =

                    data = Data(x=x, edge_index=edge_index, y=y)
                    data.rmsd = torch.tensor([rmsds[idx]], dtype=torch.float)
                    data.energy = torch.tensor([energies[idx]], dtype=torch.float)
                    data.dist = dist

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)
                
                    pbar.update(1)
            
            pbar.close()
            torch.save(self.collate(data_list),
                       os.path.join(self.processed_dir, f'{split}.pt'))



class PDBBind_test(InMemoryDataset):

    def __init__(self, root, subset=False, split='test', transform=None,
                 pre_transform=None, pre_filter=None, protein = None):
        self.subset = subset
        self.protein = protein
        self.protein_dir = os.path.join(root, self.protein)
        # super(PDBBind, self).__init__(root, transform, pre_transform, pre_filter)
        super().__init__(root, transform, pre_transform, pre_filter)
        path = os.path.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [
            # 'train', 'test'
            'test'
        ]

    @property
    def processed_dir(self):
        # name = 'subset' if self.subset else 'full'
        # return os.path.join(self.root, name, 'processed')
        return os.path.join(self.protein_dir, 'processed')

    @property
    def processed_file_names(self):
        # return ['train.pt', 'test.pt']
        return ['test.pt']

    # TODO: implement after uploading the entire dataset somewhere
    def download(self):
        # print('Hello Download')
        pass

    def process(self):

        dataset_dir = self.protein_dir
        files_num = 1

        data_list = []
        graph_idx = 0

        pbar = tqdm(total=files_num)
        pbar.set_description(f'Processing {self.protein} dataset')
        

        with open(os.path.join(dataset_dir, f'{self.protein}_data-G.json')) as gf:
            graphs = gf.readlines()
        num_graphs_per_file = len(graphs)//3

        pbar.total = num_graphs_per_file * files_num
        pbar.refresh()

        # features = []
        # with open(os.path.join(dataset_dir, f'{f}_data-feats'), 'rb') as ff:
        #     for _ in range(num_graphs_per_file):
        #         features.append(np.load(ff))

        feat_file = open(os.path.join(dataset_dir, f'{self.protein}_data-feats'), 'rb')

        labels = np.loadtxt(os.path.join(dataset_dir, f'{self.protein}_label'))
        rmsds = labels[:, 1]
        energies = labels[:, 2]
        labels = labels[:, 0]

        for idx in range(num_graphs_per_file):

            # it's slow because of these!
            features = np.load(feat_file)
            indptr = ast.literal_eval(graphs[3*idx])
            indices = ast.literal_eval(graphs[3*idx+1])
            dist = ast.literal_eval(graphs[3*idx+2])
            
            # using scipy, probably slow
            # value = np.ones_like(indices)
            # # add 0 to indptr for scipy csr
            # num_nodes = len(indptr)
            # indptr.insert(0, 0)
            # coo_graph = csr_matrix((value, indices, indptr), shape=(
            #     num_nodes, num_nodes)).tocoo()
            # per graph data
            # edge_index = torch.Tensor(
            #     [coo_graph.row, coo_graph.col]).to(torch.long)


            indptr = torch.LongTensor(indptr)
            indices = torch.LongTensor(indices)
            dist = torch.tensor(dist, dtype=torch.float)
            row_idx = torch.ops.torch_sparse.ind2ptr(indptr,len(indices))[1:]
            edge_index = torch.stack((row_idx, indices), dim=0)
            
            # No edge attr to save space
            # edge_attr = torch.Tensor(coo_graph.data).to(torch.long)
            
            # x = torch.Tensor(features[idx])
            x = torch.Tensor(features)
            y = torch.LongTensor([labels[idx].astype(int)])
            dist = dist.reshape(dist.size()[0], 1)
            # e =

            data = Data(x=x, edge_index=edge_index, y=y)
            data.rmsd = torch.tensor([rmsds[idx]], dtype=torch.float)
            data.energy = torch.tensor([energies[idx]], dtype=torch.float)
            data.dist = dist

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        
            pbar.update(1)
        
        pbar.close()
        torch.save(self.collate(data_list),
                   os.path.join(self.processed_dir, 'test.pt'))
