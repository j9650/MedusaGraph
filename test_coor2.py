
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from scipy.spatial import distance
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn import metrics

import sys
import os
import argparse
import math
import numpy as np
from time import time
from tqdm import tqdm


from dataset import PDBBindCoor
from model import loss_fn_kd, get_soft_label, loss_fn_dir, loss_fn_cos
from molecular_optimization import get_refined_pose_file
import plot

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", help="which model we use", type=str, default='Net_coor_res')
parser.add_argument("--loss", help="which loss function we use", type=str, default='L1Loss')
parser.add_argument("--loss_reduction", help="reduction approach for loss function", type=str, default='mean')
parser.add_argument("--lr", help="learning rate", type=float, default = 0.0001)
parser.add_argument("--epoch", help="epoch", type=int, default = 1000)
parser.add_argument("--start_epoch", help="epoch", type=int, default = 1)
parser.add_argument("--batch_size", help="batch_size", type=int, default = 64)
parser.add_argument("--gpu_id", help="id of gpu", type=int, default = 3)
parser.add_argument("--data_path", help="train keys", type=str, default='/gpfs/group/mtk2/cyberstar/hzj5142/GNN/GNN/DGNN/data/pdbbind/pdbbind_rmsd_srand200/')
parser.add_argument("--heads", help="number of heads for multi-attention", type=int, default = 1)
parser.add_argument("--edge_dim", help="dimension of edge feature", type=int, default = 3)
parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default = 1)
parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default = 256)
parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default = 0)
parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default = 512)
parser.add_argument("--output", help="train result", type=str, default='output_train')
parser.add_argument("--model_dir", help="save best model", type=str, default='best_model.pt')
parser.add_argument("--pre_model", help="pre trained model", type=str, default='None')
parser.add_argument("--th", help="threshold for positive pose", type=float, default=3.00)
parser.add_argument("--dropout_rate", help="dropout rate", type=float, default=0.3)
parser.add_argument("--weight_bias", help="weight bias", type=float, default=1.0)
parser.add_argument("--last", help="activation of last layer", type=str, default='log')
parser.add_argument("--KD", help="if we apply knowledge distillation (Yes / No)", type=str, default='No')
parser.add_argument("--KD_soft", help="function convert rmsd to softlabel", type=str, default='exp')
parser.add_argument("--edge", help="if we use edge attr", type=bool, default=False)
parser.add_argument("--plt_dir", help="path to the plot figure", type=str, default='best_model_plt')
parser.add_argument("--flexible", help="if we only calculate flexible nodes", default=False, action='store_true')
parser.add_argument("--residue", help="if we apply residue connection to CONV layers", default=False, action='store_true')
parser.add_argument("--iterative", help="if we iteratively calculate the pose", type=int, default = 0)
parser.add_argument("--pose_limit", help="maximum poses to be evaluated", type=int, default = 0)
parser.add_argument("--step_len", help="length of the moving vector", type=float, default = 0.03)
parser.add_argument("--class_dir", help="classify the direction on each axis", default=False, action='store_true')
parser.add_argument("--max_atom", help="we only test the pose with flexible atoms less than max_atom", type=int, default = 0)
parser.add_argument("--max_rotamer", help="we only test the pose with rotatable bonds less than max_rotamer", type=int, default = 0)
parser.add_argument("--fingerprint_path", help="path to the finger print", type=str, default='best_model_plt')
parser.add_argument("--tot_seed", help="num of seeds in the dataset", type=int, default = 8)

args = parser.parse_args()
print(args)

path = args.data_path

train_datasets = []
test_datasets = []
train_loaders = []
test_loaders = []

test_dataset=PDBBindCoor(root=path, split='test', data_type='autodock')
test_loader=DataLoader(test_dataset, batch_size=1)


test_dataset_size = len(test_dataset)
test_loader_size = len(test_loader.dataset)
print(f"test_dataset_size: {test_dataset_size}, test_loader_size: {test_loader_size}")

weight = 4.100135326385498 + args.weight_bias
print(f"weight: 1:{weight}")




def _dir_2_coor(out, length):
    out = out.exp()
    x = out[:, 4:8].sum(1) - out[:, :4].sum(1)
    y = out[:,[2,3,6,7]].sum(1) - out[:,[0,1,4,5]].sum(1)
    z = out[:,[1,3,5,7]].sum(1) - out[:,[0,2,4,6]].sum(1)
    # ans = torch.stack([out[:, 1] - out[:, 0], out[:, 3] - out[:, 2], out[:, 5] - out[:, 4]], 1)
    ans = torch.stack([x, y, z], 1)

    return ans*length


gpu_id = str(args.gpu_id)
device_str = 'cuda:' + gpu_id if torch.cuda.is_available() else 'cpu'

device = torch.device(device_str)
print('cuda' if torch.cuda.is_available() else 'cpu')

from model import Net_coor, Net_coor_res, Net_coor_dir, Net_coor_len, Net_coor_cent

if args.model_type == 'Net_coor_res':
    model = Net_coor_res(test_dataset.num_features, args).to(device)
elif args.model_type == 'Net_coor':
    model = Net_coor(test_dataset.num_features, args).to(device)
elif args.model_type == 'Net_coor_dir':
    model = Net_coor_dir(test_dataset.num_features, args).to(device)
elif args.model_type == 'Net_coor_len':
    model = Net_coor_len(test_dataset.num_features, args).to(device)
    # assert args.class_dir
elif args.model_type == 'Net_coor_cent':
    model = Net_coor_cent(test_dataset.num_features, args).to(device)

if args.pre_model != 'None':
    model = torch.load(args.pre_model).to(device)

if args.loss == 'L1Loss':
    loss_op = torch.nn.L1Loss(reduction=args.loss_reduction)
elif args.loss == 'MSELoss':
    loss_op = torch.nn.MSELoss(reduction=args.loss_reduction)
elif args.loss == 'CosineEmbeddingLoss':
    loss_op = torch.nn.CosineEmbeddingLoss(reduction=args.loss_reduction)
    cos_target = torch.tensor([1]).to(device)
elif args.loss == 'CosineAngle':
    loss_op = loss_fn_cos(device, reduction=args.loss_reduction)
    loss_op2 = torch.nn.CosineEmbeddingLoss(reduction=args.loss_reduction)
    cos_target = torch.tensor([1]).to(device)
if args.class_dir:
    loss_op = torch.nn.CrossEntropyLoss()
    assert args.model_type == 'Net_coor_dir'
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


@torch.no_grad()
def test(loader, epoch):
    model.eval()
    t = time()

    total_loss = 0
    total_rmsd = 0.0
    total_rmsd_in = 0.0
    all_rmsds = []
    all_rmsds_in = []
    total_atoms = 0
    rmsd_decreased = 0

    pose_idx = 0
    gstd = 0
    total_rmsds = [0.0 for i in range(args.iterative)]
    avg_rmsd = 0.0
    fp = 0
    tp = 0
    fn = 0
    tn = 0
    fpl = [0 for i in range(8)]
    good_pose = 0
    good_pose_in = 0

    all_atoms = 0
    ligand_atoms = 0
    diff_complex = 0
    diff_complexes = []
    rotamer_list = []

    rmsd_per_pdb = {}
    num_pose_per_pdb = {}
    rmsd_per_pdb_in = {}
    best_rmsd_per_pdb = {}
    best_rmsd_per_pdb_in = {}
    pdb = ''

    if args.max_rotamer > 0:
        with open(args.fingerprint_path, 'r') as f:
            for line in f:
                rotamer_list.append(int(line.strip().split()[-1]))
    
    pbar = tqdm(total=test_loader_size)
    pbar.set_description('Testing poses...')
    for data in loader:
        num_atoms = data.x.size()[0]
        num_flexible_atoms = data.x[data.flexible_idx.bool()].size()[0]


        pdb = str(data.pdb[0])

        if pdb not in rmsd_per_pdb:
            diff_complex += 1
            all_atoms = num_atoms
            ligand_atoms = num_flexible_atoms
            rmsd_per_pdb[pdb] = 0.0
            rmsd_per_pdb_in[pdb] = 0.0
            num_pose_per_pdb[pdb] = 0.0
            best_rmsd_per_pdb[pdb] = 9999.0
            best_rmsd_per_pdb_in[pdb] = 9999.0


        if args.max_rotamer > 0:
            if args.max_rotamer < rotamer_list[(diff_complex - 1) % 751]:
                pbar.update(1)
                continue

        if args.max_atom > 0 and num_flexible_atoms > args.max_atom:
            pbar.update(1)
            continue
        if data.x.size()[0] != num_atoms:
            print(f"num_flexible_atoms: {num_flexible_atoms}, data.x.size: {data.x.size()[0]}, data.y.size: {num_atoms}")
        if args.flexible:
            if args.model_type != 'Net_coor_cent':
                out = model(data.x.to(device), data.edge_index.to(device), data.dist.to(device))[data.flexible_idx.bool()]

            if args.class_dir:
                y = data.y[data.flexible_idx.bool()].gt(0).long().to(device)
                y = y[:, 0] * 4 + y[:, 1] * 2 + y[:, 2]
                loss = loss_op(out, y)
                for i in range(8):
                    fpl[i] += y.eq(i).sum().cpu().item()

                tn += y.size()[0]
                out = _dir_2_coor(out, args.step_len)
            elif args.loss == 'CosineEmbeddingLoss':
                loss = loss_op(out, data.y.to(device)[data.flexible_idx.bool()], cos_target)
            elif args.loss == 'CosineAngle':
                loss = (1 - loss_op2(out, data.y.to(device)[data.flexible_idx.bool()], cos_target)).acos().sum()
            elif args.model_type == 'Net_coor_len':
                length = data.y.to(device)[data.flexible_idx.bool()].square().sum(1).sqrt().reshape(out.size()[0],1)
                loss = loss_op(out, length)
                out = data.y.to(device)[data.flexible_idx.bool()]
            elif args.model_type == 'Net_coor_cent':
                pred = model(data.x.to(device), data.edge_index.to(device), data.dist.to(device), data.batch.to(device), data.flexible_idx.bool().to(device)).cpu()
                y = global_mean_pool(data.y[data.flexible_idx.bool()], data.batch[data.flexible_idx.bool()])
                loss = loss_op(pred, y)
                out = pred.repeat(num_flexible_atoms, 1)
            else:
                loss = loss_op(out, data.y.to(device))
            
            this_rmsd = (F.mse_loss(data.y.to(device), out, reduction='sum') / num_flexible_atoms).sqrt().cpu().item()
            total_rmsd += this_rmsd
            all_rmsds.append(this_rmsd)

            num_pose_per_pdb[pdb] += 1
            rmsd_per_pdb[pdb] += this_rmsd
            if this_rmsd < best_rmsd_per_pdb[pdb]:
                best_rmsd_per_pdb[pdb] = this_rmsd

        else: # not flexible
            out = model(data.x.to(device), data.edge_index.to(device), data.dist.to(device))
            loss = loss_op(out, data.y.to(device))

            rmsds = F.mse_loss(data.y[data.flexible_idx.bool()], out.cpu()[data.flexible_idx.bool()], reduction='sum').item()
            total_rmsd += math.sqrt(rmsds / num_flexible_atoms)
            all_rmsds.append(math.sqrt(rmsds / num_flexible_atoms))


        if (epoch <= 1):
            if args.flexible:
                rmsds = math.sqrt(torch.sum(torch.square(data.y)).item() / num_flexible_atoms)
                total_rmsd_in += rmsds
                all_rmsds_in.append(rmsds)
                rmsd_decreased += 1 if this_rmsd < rmsds else 0
                if rmsds < args.th:
                    good_pose_in += 1
                if this_rmsd < args.th:
                    good_pose += 1
                rmsd_per_pdb_in[pdb] += rmsds
                if rmsds < best_rmsd_per_pdb_in[pdb]:
                    best_rmsd_per_pdb_in[pdb] = rmsds

            else:
                rmsds = F.mse_loss(data.y, data.x[:, -3:], reduction='sum').item()
                total_rmsd_in += math.sqrt(rmsds / num_atoms)
                all_rmsds_in.append(math.sqrt(rmsds / num_atoms))
        # all_rmsds_in = all_rmsds_in + rmsds

        if args.loss == 'CosineEmbeddingLoss':
            total_loss += loss.item() / num_flexible_atoms
        elif args.loss == 'CosineAngle':
            total_loss += loss.item() / num_flexible_atoms
        else:
            total_loss += loss.item() # * args.batch_size

        pose_idx += 1
        pbar.update(1)
        if args.pose_limit > 0 and pose_idx >= args.pose_limit:
            break

    pdb_key = [k for k in rmsd_per_pdb]
    assert diff_complex == len(pdb_key)
    print(f'diff_complex {diff_complex}')
    succ_pdb = len([1 for k in pdb_key if best_rmsd_per_pdb[k] <= args.th])
    succ_pdb_in = len([1 for k in pdb_key if best_rmsd_per_pdb_in[k] <= args.th])
    print(f'succ_pdb {succ_pdb} succ_pdb_in {succ_pdb_in}')
    avg_rmsd_per_pdb = sum(rmsd_per_pdb[k] / num_pose_per_pdb[k] for k in pdb_key) / diff_complex
    avg_rmsd_per_pdb_in = sum(rmsd_per_pdb_in[k] / num_pose_per_pdb[k] for k in pdb_key) / diff_complex

    tt = time() - t
    pbar.close()
    plot.rmsd_hist(np.array(all_rmsds)*100, os.path.join(args.plt_dir, f"rmsd_hist_final"), xlim=20, title='Poses Genrated by Pose-prediction Model')
    plot.rmsd_hist(np.array(all_rmsds_in)*100, os.path.join(args.plt_dir, f"rmsd_hist_init"), xlim=20, title='Poses Genrated by MedusaDock')
    print(f'good pose: {good_pose} good_pose_in: {good_pose_in}')
    std_all_in = np.std(all_rmsds_in)*100
    std_all = np.std(all_rmsds)*100
    print(f'std: {std_all_in} {std_all}')

    print(f"Spend {tt}s")
    print(f"x: {fpl}, all: {tn}")
    print([i / pose_idx for i in total_rmsds])
    print(avg_rmsd / pose_idx)
    print(f'total poses: {pose_idx}, {rmsd_decreased} poses decreased rmsd')
    if args.output != 'output_train':
        with open(args.output, 'a') as f:
            if args.max_atom != 0:
                f.write(f'{args.max_atom} {total_rmsd_in / pose_idx} {total_rmsd / pose_idx} {pose_idx} {good_pose_in} {good_pose} {rmsd_decreased} {std_all_in} {std_all}\n')
            elif args.max_rotamer != 0:
                f.write(f'{args.max_rotamer} {total_rmsd_in / pose_idx} {total_rmsd / pose_idx} {pose_idx} {good_pose_in} {good_pose} {rmsd_decreased} {std_all_in} {std_all}\n')
    return total_loss / pose_idx, avg_rmsd_per_pdb * 100, avg_rmsd_per_pdb_in * 100


if not os.path.isdir(args.model_dir):
    os.makedirs(args.model_dir)
if not os.path.isdir(args.plt_dir):
    os.makedirs(args.plt_dir)
min_rmsd = 10.0
best_epoch = 0
epoch = 1
loss, rmsd, rmsd_in = test(test_loader, epoch)
print(f"Epoch: {epoch} Test Loss: {loss}  Avg RMSD: {rmsd}")
print(f"Avg RMSD of inputs: {rmsd_in}")
