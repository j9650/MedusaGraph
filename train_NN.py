
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.metrics import f1_score
from sklearn import metrics

import sys
import os

import argparse

# from torch_geometric.datasets import PPI
# path = '/export/local/mfr5226/datasets/pyg/ppi'
# train_dataset = PPI(path, split='train')
# test_dataset = PPI(path, split='test')

from dataset import PDBBind, PDBBind_test
from model import Net, Net_en
import numpy as np
import math


def find_backup_pose(labels, scores, fix_th, gt_th):
    zipped = zip(labels, scores)
    zipped_tmp = sorted(zipped, key=lambda x: x[1], reverse=True)
    succ = 0
    l = [zipped_tmp[i][0] for i in range(fix_th)]
    for x in l:
        if x <= gt_th:
            succ += 1

    return l, succ


def avg_mean(iterable):
    a = np.array(iterable)
    return np.sum(a) / len(a)

parser = argparse.ArgumentParser()
parser.add_argument("--lr", help="learning rate", type=float, default = 0.0001)
parser.add_argument("--epoch", help="epoch", type=int, default = 1000)
parser.add_argument("--gpu_id", help="id of gpu", type=int, default = 3)
parser.add_argument("--data_path", help="path to the pdb data", type=str, default='/home/mdl/hzj5142/GNN/pdb-gnn/data/pdbbind_rmsd_srand4')
parser.add_argument("--pdb_list", help="list of proteins", type=str, default='/home/mdl/hzj5142/GNN/pdb-gnn/pdbbind_listtest50-100')
parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default = 1)
parser.add_argument("--d_graph_layer", help="number of GNN layer", type=int, default = 256)
parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default = 0)
parser.add_argument("--d_FC_layer", help="number of FC layer", type=int, default = 512)
parser.add_argument("--output", help="train result", type=str, default='output_train')
parser.add_argument("--model_dir", help="save best model", type=str, default='best_model.pt')
parser.add_argument("--output_dir", help="save best model", type=str, default='best_model.pt')
parser.add_argument("--dropout_rate", help="dropout rate", type=float, default=0.3)
parser.add_argument("--rmsd_th", help="rmsd threshold for good/bad pose", type=str, default='3.00')
parser.add_argument("--select_num", help="number of poses select for each protein", type=str, default = '08')
parser.add_argument("--start", help="number of poses select for each protein", type=int, default = 15)
parser.add_argument("--end", help="number of poses select for each protein", type=int, default = 200)
parser.add_argument("--step", help="number of poses select for each protein", type=int, default = 5)
parser.add_argument("--edge", help="has edge features or not", type=bool, default = False)
args = parser.parse_args()
print(args)

gpu_id = str(args.gpu_id)
device_str = 'cuda:' + gpu_id if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)
print('cuda' if torch.cuda.is_available() else 'cpu')



@torch.no_grad()
def test(it, rmsd_th_str = '3.00', select_num_str = '08'):
    rmsd_th = float(rmsd_th_str)
    select_num = int(select_num_str)
    model_dir = os.path.join(args.model_dir, f'model_iter{it}.pt')
    print(f"Before Load model{model_dir}, {device_str}")
    model = torch.load(model_dir, map_location=device_str).to(device)

    print("Before model eval")
    model.eval()

    output_dir = os.path.join(args.output_dir, f't{select_num_str}_{rmsd_th_str}_iter{it}.txt')
    f_out = open(output_dir, "w")

    print("Load model done")

    proteins = []
    with open(args.pdb_list,"r") as f:
        for line in f:
            proteins.append(line.strip())

    success_pdb = 0
    stop_attemp_list = []
    candidate_pose = 0
    succ_candidates = 0
    tot_attempts = 0
    for protein in proteins:
        dataset = PDBBind_test(root=args.data_path, split='test', protein = protein)
        loader = DataLoader(dataset, batch_size=1)
        # continue
        flag = 0
        pose_id = 0
        seed = 0
        index = []
        success_poses = 0
        selected_pose = 0
        rmsds = []
        with open(os.path.join(args.data_path, protein, protein+"_index"), "r") as f:
            for line in f:
                index.append(int(line))

        stop_seed = 1993
        backup_rmsd = []
        backup_pred = []
        for data in loader:
            if not args.edge:
                out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device), energy = data.energy.view(data.energy.size()[0], 1).to(device))
            else:
                out = model(data.x.to(device), data.edge_index.to(device), data.dist.to(device), data.batch.to(device), energy = data.energy.view(data.energy.size()[0], 1).to(device))
            pred = math.exp(out.cpu().numpy()[0, 1])
            if pred > 0.5:
                selected_pose += 1
                this_rmsd = data.rmsd.numpy()[0]
                rmsds.append(this_rmsd)
                if this_rmsd <= rmsd_th:
                    success_poses += 1
            else:
                backup_rmsd.append(data.rmsd.numpy()[0])
                backup_pred.append(pred)
            pose_id += 1
            if pose_id >= index[seed]:
                seed += 1
                pose_id = 0
                if selected_pose >= select_num:
                    stop_seed = seed
                    break

        stop_attemp_list.append(stop_seed)
        if len(rmsds) < select_num:
            l, n = find_backup_pose(backup_rmsd, backup_pred, select_num - len(rmsds), rmsd_th)
            rmsds += l
            success_poses += n
        avg_rmsd = avg_mean(rmsds)
        # print(rmsds)

        if stop_seed < 1993 or success_poses > 0:
            candidate_pose += select_num
            succ_candidates += success_poses
            tot_attempts += stop_seed
        print(f"{protein} found {success_poses} good poses, stop at {stop_seed}, avg rmsd is {avg_rmsd}")
        f_out.write(f"{protein} found {success_poses} good poses, stop at {stop_seed}, avg rmsd is {avg_rmsd}\n")
        if success_poses > 0:
            success_pdb += 1
        #break

    print(str(success_pdb)+' proteins success! ' + str(succ_candidates) + ' good pose')
    f_out.write(str(success_pdb)+' proteins success! ' + str(succ_candidates) + ' good pose\n')
    #print('average attemps: '+str(geo_mean(stop_attemp_list)))
    print('average attemps: '+str(avg_mean(stop_attemp_list)))
    print('total_attemps: '+str(tot_attempts)+' attemps_per_succ_pose: '+str(tot_attempts/succ_candidates)+" accuracy: "+str(succ_candidates/candidate_pose))
    f_out.write('average attemps: '+str(avg_mean(stop_attemp_list)) + '\n')
    f_out.write('total_attemps: '+str(tot_attempts)+' attemps_per_succ_pose: '+str(tot_attempts/succ_candidates)+" accuracy: "+str(succ_candidates/candidate_pose))
    f_out.close()

for it in range(args.start, args.end, args.step):
    # print(f"evaluate iter {it}")
    test(it = it, rmsd_th_str=args.rmsd_th, select_num_str=args.select_num)
    # break

