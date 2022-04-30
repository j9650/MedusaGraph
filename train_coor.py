
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

# from torch_geometric.datasets import PPI
# path = '/export/local/mfr5226/datasets/pyg/ppi'
# train_dataset = PPI(path, split='train')
# test_dataset = PPI(path, split='test')

from dataset import PDBBindCoor
from model import loss_fn_kd, get_soft_label, loss_fn_dir, loss_fn_cos
import plot

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", help="which model we use", type=str, default='Net_coor_res')
parser.add_argument("--loss", help="which loss function we use", type=str, default='L1Loss')
parser.add_argument("--loss_reduction", help="reduction approach for loss function", type=str, default='mean')
parser.add_argument("--lr", help="learning rate", type=float, default = 0.0001)
parser.add_argument("--epoch", help="epoch", type=int, default = 1000)
parser.add_argument("--start_epoch", help="epoch", type=int, default = 1)
parser.add_argument("--batch_size", help="batch_size", type=int, default = 64)
parser.add_argument("--atomwise", help="if we train the model atomwisely", type=int, default = 0)
parser.add_argument("--gpu_id", help="id of gpu", type=int, default = 3)
parser.add_argument("--data_path", help="train keys", type=str, default='/gpfs/group/mtk2/cyberstar/hzj5142/GNN/GNN/DGNN/data/pdbbind/pdbbind_rmsd_srand200/')
parser.add_argument("--heads", help="number of heads for multi-attention", type=int, default = 1)
parser.add_argument("--edge_dim", help="dimension of edge feature", type=int, default = 3)
parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default = 1)
parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default = 256)
parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default = 0)
parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default = 512)
parser.add_argument("--output", help="train result", type=str, default='none')
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
parser.add_argument("--hinge", help="rate of hinge loss", type=float, default = 0)
parser.add_argument("--tot_seed", help="num of seeds in the dataset", type=int, default = 8)

args = parser.parse_args()
print(args)

if args.atomwise:
    args.batch_size = 1


#path = '/export/local/mfr5226/datasets/pdb/sample/'

# path = '/home/mdl/hzj5142/GNN/pdb-gnn/data/pdbbind/'
# path = '/home/mdl/hzj5142/GNN/pdb-gnn/data/pdbbind_rmsd_srand4'
path = args.data_path

train_datasets = []
test_datasets = []
train_loaders = []
test_loaders = []

train_dataset=PDBBindCoor(root=path, split='train')
test_dataset=PDBBindCoor(root=path, split='test')
train_loader=DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=1)


train_loader_size = len(train_loader.dataset)
test_dataset_size = len(test_dataset)
test_loader_size = len(test_loader.dataset)

print(f"total {len(train_datasets)} subdatasets")
print(f"train_loader_size: {train_loader_size}")
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
# model = Net(train_datasets[0].num_features, train_datasets[0].num_classes, args).to(device)

from model import Net_coor, Net_coor_res, Net_coor_dir, Net_coor_len, Net_coor_cent

if args.model_type == 'Net_coor_res':
    model = Net_coor_res(train_dataset.num_features, args).to(device)
elif args.model_type == 'Net_coor':
    model = Net_coor(train_dataset.num_features, args).to(device)
elif args.model_type == 'Net_coor_dir':
    model = Net_coor_dir(train_dataset.num_features, args).to(device)
elif args.model_type == 'Net_coor_len':
    model = Net_coor_len(train_dataset.num_features, args).to(device)
    # assert args.class_dir
elif args.model_type == 'Net_coor_cent':
    model = Net_coor_cent(train_dataset.num_features, args).to(device)

if args.pre_model != 'None':
    model = torch.load(args.pre_model, map_location=device_str).to(device)
# loss_op = torch.nn.MSELoss()

# model.double()
# torch.set_default_dtype(torch.float64)

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
    # loss_op = torch.nn.CosineEmbeddingLoss(reduction='none')
    cos_target = torch.tensor([1]).to(device)
if args.class_dir:
    # loss_op = loss_fn_dir(device)
    loss_op = torch.nn.CrossEntropyLoss()
    assert args.model_type == 'Net_coor_dir'
# loss_op_kld = torch.nn.KLDivLoss()
hinge = torch.tensor([args.hinge]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def bond_dist(data, pred, fix_idx):
    # print(data.x.size(), data.edge_index.size(), pred.size())
    x = data.edge_index[0, fix_idx]
    y = data.edge_index[1, fix_idx]
    # print(x, y)
    # print(data.flexible_idx)
    # print(fix_idx)
    # print(data.edge_index[:, fix_idx])
    # print(data.dist)
    #print(data.edge_index[1, fix_idx])
    #print(fix_idx)
    node_x = data.x[x, -3:] + pred[x]
    node_y = data.x[y, -3:] + pred[y]
    # dist = (node_x - node_y).square()
    dist = torch.nn.MSELoss(reduction='none')(node_x, node_y)

    return dist.sum(-1).sqrt()

def train():
    model.train()

    total_loss = 0
    tot = 0
    t = time()
    pbar = tqdm(total=train_loader_size)
    pbar.set_description('Training poses...')
    for data in train_loader:
        with torch.cuda.amp.autocast():
            data = data.to(device)
            # print(data.x.size(), data.y.size(), data.edge_index.size(), data.dist.size())
            # print(type(data.flexible_idx.bool()))
            # print(data.flexible_idx.bool())
            # print(data.flexible_idx.bool().size(), data.x[data.flexible_idx.bool()].size())
            # print(data.edge_index)
            if args.atomwise:
                flexible_len = data.flexible_len.cpu().item()
                all_atom_idx = torch.randperm(flexible_len)
                # iters = flexible_len if flexible_len < args.atomwise else args.atomwise
                avg_loss = 0.0
                for idx in range(args.atomwise):
                    st = (idx * flexible_len) // args.atomwise
                    ed = ((idx + 1) * flexible_len) // args.atomwise
                    atom_idx = all_atom_idx[st:ed]
                    optimizer.zero_grad()
                    pred = model(data.x, data.edge_index, data.dist)[atom_idx]
                    loss = loss_op(pred, data.y[atom_idx])
                    avg_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                total_loss += avg_loss / args.atomwise
                tot += 1
                pbar.update(1)
                continue

            optimizer.zero_grad()
            if args.flexible:

                if args.model_type != 'Net_coor_cent':
                    pred = model(data.x, data.edge_index, data.dist)[data.flexible_idx.bool()]
                # length = pred.square().sum(1).sqrt()
                # pred = pred.transpose(0,1).div(length).transpose(0, 1) * args.step_len
                # print(pred.size())
                if args.class_dir:
                    y = data.y[data.flexible_idx.bool()].gt(0).long()
                    y = y[:, 0] * 4 + y[:, 1] * 2 + y[:, 2]
                    loss = loss_op(pred, y)
                elif args.loss == 'CosineEmbeddingLoss':
                    loss = loss_op(pred, data.y[data.flexible_idx.bool()], cos_target)
                elif args.loss == 'CosineAngle':
                    loss = loss_op(pred, data.y[data.flexible_idx.bool()])
                    # loss = (1 - loss_op(pred, data.y[data.flexible_idx.bool()], cos_target)).acos().sum()
                elif args.model_type == 'Net_coor_len':
                    length = data.y[data.flexible_idx.bool()].square().sum(1).sqrt().reshape(pred.size()[0],1)
                    loss = loss_op(pred, length)
                elif args.model_type == 'Net_coor_cent':
                    pred = model(data.x, data.edge_index, data.dist, data.batch, data.flexible_idx.bool())
                    y = global_mean_pool(data.y[data.flexible_idx.bool()], data.batch[data.flexible_idx.bool()])
                    loss = loss_op(pred, y)
                elif args.hinge != 0:
                    # l1 = (data.flexible_idx[data.edge_index[0]]==0).nonzero(as_tuple=True)[0].cpu().numpy()
                    # l2 = (data.flexible_idx[data.edge_index[1]]==0).nonzero(as_tuple=True)[0].cpu().numpy()
                    # fix_idx = np.intersect1d(l1, l2)
                    fix_idx = (data.dist[:, 0] != 0).nonzero(as_tuple=True)[0] # .cpu().numpy()
                    # print(fix_idx.size())
                    # loss1 = loss_op(bond_dist(data, pred, fix_idx), data.dist[fix_idx, 0])
                    # loss1 = loss_op(bond_dist(data, pred, fix_idx), bond_dist(data, data.y, fix_idx))
                    # loss1 = torch.nn.HingeEmbeddingLoss(margin=0.01)(bond_dist(data, pred, fix_idx), torch.LongTensor([-1 for _ in fix_idx]).to(device))
                    bond_diff = bond_dist(data, pred, fix_idx) - bond_dist(data, data.y, fix_idx)
                    # print(bond_diff.size(), fix_idx.size())
                    l = fix_idx.size()[0]
                    # loss1 = torch.nn.HingeEmbeddingLoss(margin=0.001)(bond_diff, torch.LongTensor([-1 for _ in fix_idx]).to(device))
                    loss1 = torch.nn.HingeEmbeddingLoss(margin=0.001)(bond_diff, torch.LongTensor([-1] * l).to(device))
                    loss = loss_op(pred, data.y) + loss1 * hinge
                else:
                    # loss = loss_op(pred, data.y[data.flexible_idx.bool()]) # + loss_op_kld(pred, data.y[data.flexible_idx.bool()])

                    loss = loss_op(pred, data.y)


            else:
                pred = model(data.x, data.edge_index, data.dist)
                # print(pred.size())
                loss = loss_op(pred, data.y) # + loss_op_kld(pred, data.y)
            if args.loss == 'CosineEmbeddingLoss':
                total_loss += loss.item() / pred.size()[0] * args.batch_size
            if args.loss == 'CosineAngle':
                total_loss += loss.item() / pred.size()[0] * args.batch_size
            else:
                total_loss += loss.item() * args.batch_size

        loss.backward()
        optimizer.step()
        # optimizer.zero_grad()
        tot += 1
        pbar.update(1)
        # break
    pbar.close()
    
    print(f"trained {tot} batches, take {time() - t}s")
    return total_loss / train_loader_size
    # return total_loss / len(train_loader.dataset)


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

    pose_idx = 0
    gstd = 0
    total_rmsds = [0.0 for i in range(args.iterative)]
    avg_rmsd = 0.0
    fp = 0
    tp = 0
    fn = 0
    tn = 0
    fpl = [0 for i in range(8)]


    all_atoms = 0
    ligand_atoms = 0
    diff_complex = 0
    rmsd_per_pdb = []
    num_pose_per_pdb = []
    rmsd_per_pdb_in = []
    pdb = ''
    pdbs = []

    pbar = tqdm(total=test_loader_size)
    pbar.set_description('Testing poses...')
    for data in loader:
        pbar.update(1)
        # with torch.cuda.amp.autocast():
        num_atoms = data.x.size()[0]
        num_flexible_atoms = data.x[data.flexible_idx.bool()].size()[0]

        # if num_atoms != all_atoms or num_flexible_atoms != ligand_atoms:
        if data.pdb != pdb:
            diff_complex += 1
            all_atoms = num_atoms
            ligand_atoms = num_flexible_atoms
            rmsd_per_pdb.append(0.0)
            rmsd_per_pdb_in.append(0.0)
            num_pose_per_pdb.append(0)
            pdb = data.pdb
            pdbs.append(pdb[0])

        if data.x.size()[0] != num_atoms:
            print(f"num_flexible_atoms: {num_flexible_atoms}, data.x.size: {data.x.size()[0]}, data.y.size: {num_atoms}")
        if args.flexible:
            # print(data.flexible_idx.bool().size())
            if args.model_type != 'Net_coor_cent':
                out = model(data.x.to(device), data.edge_index.to(device), data.dist.to(device))[data.flexible_idx.bool()]
            # length = out.square().sum(1).sqrt()
            # out = out.transpose(0,1).div(length).transpose(0, 1) * args.step_len
            if args.class_dir:
                y = data.y[data.flexible_idx.bool()].gt(0).long().to(device)
                y = y[:, 0] * 4 + y[:, 1] * 2 + y[:, 2]
                loss = loss_op(out, y)
                for i in range(8):
                    fpl[i] += y.eq(i).sum().cpu().item()

                # fp += y.eq(0).sum().cpu().item()
                # fn += y.eq(1).sum().cpu().item()
                # tp += y.eq(2).sum().cpu().item()
                tn += y.size()[0]
                out = _dir_2_coor(out, args.step_len)
            elif args.loss == 'CosineEmbeddingLoss':
                loss = loss_op(out, data.y.to(device)[data.flexible_idx.bool()], cos_target)
            elif args.loss == 'CosineAngle':
                # loss = loss_op(out, data.y.to(device)[data.flexible_idx.bool()])
                loss = (1 - loss_op2(out, data.y.to(device)[data.flexible_idx.bool()], cos_target)).acos().sum()
                # loss = (1 - loss_op(out, data.y.to(device)[data.flexible_idx.bool()], cos_target)).acos().sum()
            elif args.model_type == 'Net_coor_len':
                length = data.y.to(device)[data.flexible_idx.bool()].square().sum(1).sqrt().reshape(out.size()[0],1)
                loss = loss_op(out, length)
                out = data.y.to(device)[data.flexible_idx.bool()]
            elif args.model_type == 'Net_coor_cent':
                pred = model(data.x.to(device), data.edge_index.to(device), data.dist.to(device), data.batch.to(device), data.flexible_idx.bool().to(device)).cpu()
                y = global_mean_pool(data.y[data.flexible_idx.bool()], data.batch[data.flexible_idx.bool()])
                loss = loss_op(pred, y)
                out = pred.repeat(num_flexible_atoms, 1)
            elif args.hinge != 0:
                fix_idx = (data.dist[:, 0] != 0).nonzero(as_tuple=True)[0]
                # loss1 = torch.nn.HingeEmbeddingLoss(margin=0.01)(bond_dist(data.to(device), out, fix_idx), torch.LongTensor([-1 for _ in fix_idx]).to(device))
                bond_diff = bond_dist(data.to(device), out, fix_idx) - bond_dist(data.to(device), data.y.to(device), fix_idx)
                loss1 = torch.nn.HingeEmbeddingLoss(margin=0.001)(bond_diff, torch.LongTensor([-1 for _ in fix_idx]).to(device))
                loss = loss_op(out, data.y.to(device)) + loss1 * args.hinge

            else:
                # loss = loss_op(out, data.y.to(device)[data.flexible_idx.bool()]) # + loss_op_kld(out, data.y.to(device)[data.flexible_idx.bool()])
                loss = loss_op(out, data.y.to(device)) # + loss_op_kld(out, data.y.to(device)[data.flexible_idx.bool()])
            # print(f"flexible nodes: {num_atoms}, all nodes: {data.flexible_idx.bool().size()[0]}")
            # rmsds = [torch.dist(data.y[i], out[i].cpu(), p=2).item() for i in range(num_atoms)]
            
            # rmsds = F.mse_loss(data.y[data.flexible_idx.bool()], out.cpu(), reduction='sum').item()
            rmsds = math.sqrt(F.mse_loss(data.y.to(device), out, reduction='sum').cpu().item() / num_flexible_atoms)
            total_rmsd += rmsds
            all_rmsds.append(rmsds)

            num_pose_per_pdb[-1] += 1
            rmsd_per_pdb[-1] += rmsds

            '''
            std_x = torch.square(torch.std(out[:, 0])).cpu().item()
            std_y = torch.square(torch.std(out[:, 1])).cpu().item()
            std_z = torch.square(torch.std(out[:, 2])).cpu().item()
            y = data.y[data.flexible_idx.bool()]
            std_y_x = torch.square(torch.std(y[:, 0])).cpu().item()
            std_y_y = torch.square(torch.std(y[:, 1])).cpu().item()
            std_y_z = torch.square(torch.std(y[:, 2])).cpu().item()
            if std_x + std_y + std_z > (std_y_x + std_y_y + std_y_z) * 0.9:
                gstd += 1
            '''
            if args.iterative > 0 and epoch > 3:
                data_x = data.x.to(device)
                data_y = data.y[data.flexible_idx.bool()].to(device)
                data_x[data.flexible_idx.bool(), -3:] += out
                avg_out = out
                dist = data.dist.numpy()
                edge_index = data.edge_index.numpy()
                l1 = (data.flexible_idx[data.edge_index[0]]==1).nonzero(as_tuple=True)[0]
                l2 = (data.flexible_idx[data.edge_index[1]]==1).nonzero(as_tuple=True)[0]
                idx = torch.cat((l1, l2), 0).unique().numpy()
                x=data_x.cpu().numpy()
                for i in idx:
                    a = edge_index[0][i]
                    b = edge_index[1][i]

                    dis = distance.euclidean(x[a, -3:], x[b, -3:])
                    dis = round(dis*100000) / 100000
                    for t in range(3):
                        if data.dist[i, t] != 0:
                            data.dist[i, t] = dis
                dist = torch.Tensor(dist).to(device)
                for ii in range(args.iterative):
                    out1 = model(data_x, data.edge_index.to(device), dist)[data.flexible_idx.bool()]
                    out += out1
                    rmsds = F.mse_loss(data_y, out, reduction='sum').item()
                    total_rmsds[ii] += math.sqrt(rmsds / num_flexible_atoms)

                    data_x[data.flexible_idx.bool(), -3:] += out1
                    avg_out += out
                    if ii < args.iterative - 1:
                        dist = data.dist.numpy()
                        x=data_x.cpu().numpy()
                        for i in idx:
                            a = edge_index[0][i]
                            b = edge_index[1][i]
                            dis = distance.euclidean(x[a, -3:], x[b, -3:])
                            dis = round(dis*100000) / 100000
                            for t in range(3):
                                if data.dist[i, t] != 0:
                                    data.dist[i, t] = dis
                        dist = torch.Tensor(dist).to(device)
                avg_out = avg_out / (args.iterative + 1)
                rmsds = F.mse_loss(data_y, avg_out, reduction='sum').item()
                avg_rmsd += math.sqrt(rmsds / num_flexible_atoms)


        else: # not flexible
            out = model(data.x.to(device), data.edge_index.to(device), data.dist.to(device))
            loss = loss_op(out, data.y.to(device)) # + loss_op_kld(out, data.y.to(device))

            # rmsds = [torch.dist(data.y[i], out[i].cpu(), p=2).item() for i in range(num_atoms)]
            rmsds = F.mse_loss(data.y[data.flexible_idx.bool()], out.cpu()[data.flexible_idx.bool()], reduction='sum').item()
            total_rmsd += math.sqrt(rmsds / num_flexible_atoms)
            all_rmsds.append(math.sqrt(rmsds / num_flexible_atoms))
        # all_rmsds = all_rmsds + rmsds


        if (epoch <= 1):
            if args.flexible:
                # rmsds = F.mse_loss(data.y[data.flexible_idx.bool()], data.x[data.flexible_idx.bool()][:, -3:], reduction='sum').item()
                # rmsds = torch.sum(torch.square(data.y[data.flexible_idx.bool()])).item()
                rmsds = math.sqrt(torch.sum(torch.square(data.y)).item() / num_flexible_atoms)
                total_rmsd_in += rmsds
                all_rmsds_in.append(rmsds)
                rmsd_per_pdb_in[-1] += rmsds
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
        # total_atoms += num_atoms
        # print(out.cpu())
        # print(data.x[data.flexible_idx.bool()][:, -3:])
        # print(total_rmsd)
        # print(total_rmsd_in)

        # if pose_idx <= 0:
            # # print(torch.cat((data.x[data.flexible_idx.bool(), -3:], out.cpu(), data.y[data.flexible_idx.bool()]), 1))
            # print(torch.cat((data.x[data.flexible_idx.bool(), -3:], out.cpu(), data.y), 1))
        pose_idx += 1
        if args.pose_limit > 0 and pose_idx >= args.pose_limit:
            break
    
    pbar.close()
    # plot.rmsd_hist(all_rmsds, os.path.join(args.plt_dir, f"rmsd_hist_epoch_{epoch}"))
    tt = time() - t
    # print(f"Spend {tt}s, {gstd} poses has greater std, total {pose_idx} poses")
    print(f"Spend {tt}s")
    # print(f"x: {fp}, y: {fn}, z: {tp}, all: {tn}")
    print(f"x: {fpl}, all: {tn}")
    print([i / pose_idx for i in total_rmsds])
    print(avg_rmsd / pose_idx)

    print(f'diff_complex {diff_complex}')
    # for pp in pdbs:
    #     print(pp)
    assert diff_complex % args.tot_seed == 0
    diff_complex = diff_complex // args.tot_seed
    print(f'diff_complex {diff_complex}')
    for ii in range(1, args.tot_seed):
        for jj in range(diff_complex):
            num_pose_per_pdb[jj] += num_pose_per_pdb[ii * diff_complex + jj]
            rmsd_per_pdb[jj] += rmsd_per_pdb[ii * diff_complex + jj]
            rmsd_per_pdb_in[jj] += rmsd_per_pdb_in[ii * diff_complex + jj]

    avg_rmsd_per_pdb = sum([r / d for r, d in zip(rmsd_per_pdb[:diff_complex], num_pose_per_pdb[:diff_complex])]) / diff_complex
    avg_rmsd_per_pdb_in = sum([r / d for r, d in zip(rmsd_per_pdb_in[:diff_complex], num_pose_per_pdb[:diff_complex])]) / diff_complex
    return total_loss / pose_idx, avg_rmsd_per_pdb, avg_rmsd_per_pdb_in
    # return total_loss / pose_idx, total_rmsd / pose_idx, total_rmsd_in / pose_idx


if not os.path.isdir(args.model_dir):
    os.makedirs(args.model_dir)
if not os.path.isdir(args.plt_dir):
    os.makedirs(args.plt_dir)
min_rmsd = 10.0
best_epoch = 0
for epoch in range(args.start_epoch, args.start_epoch + args.epoch):
    loss = train()
    print(f"Train Loss: {loss}")
    loss, rmsd, rmsd_in = test(test_loader, epoch)
    print(f"Epoch: {epoch} Test Loss: {loss}  Avg RMSD: {rmsd}")
    if epoch <= 1:
        print(f"Avg RMSD of inputs: {rmsd_in}")
        if args.output != 'none':
            with open(args.output, 'a') as f:
                f.write(f"Avg RMSD of inputs: {rmsd_in}\n")
    if args.output != 'none':
        with open(args.output, 'a') as f:
            f.write(f"Epoch: {epoch} Test Loss: {loss}  Avg RMSD: {rmsd}\n")
    if epoch > 3 and (min_rmsd > rmsd or epoch % 5 == 0):
        saved_model_dir = os.path.join(args.model_dir, f'model_{epoch}.pt')
        torch.save(model, saved_model_dir)
        os.system(f'chmod 777 {saved_model_dir}')
        print(f"save model at epoch {epoch}, rmsd of {rmsd*100} !!!!!!!!")
        if args.output != 'none':
            with open(args.output, 'a') as f:
                f.write(f"save model at epoch {epoch}, rmsd of {rmsd*100} !!!!!!!!\n")
        if min_rmsd > rmsd:
            min_rmsd = rmsd
            best_epoch = epoch
    print("")
    os.system(f'chmod 777 {args.output}')

print(f"\nBest model at epoch {best_epoch}, rmsd is {min_rmsd}")