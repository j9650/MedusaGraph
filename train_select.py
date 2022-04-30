
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

TH = 0.03

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
parser.add_argument("--initial_mu", help="initial value of mu", type=float, default = 4.0)
parser.add_argument("--initial_dev", help="initial value of dev", type=float, default = 1.0)
parser.add_argument("--output", help="train result", type=str, default='none')
parser.add_argument("--model_dir", help="save best model", type=str, default='best_model.pt')
parser.add_argument("--pre_model", help="pre trained model", type=str, default='None')
parser.add_argument("--th", help="threshold for positive pose", type=float, default=0.025)
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
parser.add_argument("--test_mode", help="whether we only test the model", default=False, action='store_true')

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

if not args.test_mode:
    train_dataset=PDBBindCoor(root=path, split='train')
    train_loader=DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
else:
    train_dataset=PDBBindCoor(root=path, split='test')
test_dataset=PDBBindCoor(root=path, split='test')
test_loader=DataLoader(test_dataset, batch_size=1)


if not args.test_mode:
    train_loader_size = len(train_loader.dataset)
    print(f"train_loader_size: {train_loader_size}")
test_dataset_size = len(test_dataset)
test_loader_size = len(test_loader.dataset)

print(f"total {len(train_datasets)} subdatasets")
print(f"test_dataset_size: {test_dataset_size}, test_loader_size: {test_loader_size}")


def get_weight_from_dataset(train_loader, device):
    positive = 0
    negative = 0
    for data in train_loader:
        data = data.to(device)
        rmsd = data.y.square().sum(dim=1)
        rmsd = global_mean_pool(rmsd, data.batch[data.flexible_idx.bool()]).sqrt()
        data_y = (rmsd < args.th).long()
        positive += torch.count_nonzero(data_y).item()
        negative += data_y.size()[0] - torch.count_nonzero(data_y).item()

    return negative / positive


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


# weight = 4.100135326385498 + args.weight_bias
# weight = 24.49264 + args.weight_bias
# weight = 11.339606567534076 + args.weight_bias
# weight = 5.92674550039127 + args.weight_bias
if not args.test_mode:
    weight = get_weight_from_dataset(train_loader, device) + args.weight_bias
else:
    weight = 11.339606567534076 + args.weight_bias
print(f"weight: 1:{weight}")

from model import Net_coor, Net_coor_res, Net_coor_dir, Net_coor_len, Net_coor_cent, Net_screen, Net_screen_DTI, Net_screen_energy
from GNN_DTI.gnn import gnn as Net_graph_DTI

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
elif args.model_type == 'Net_screen':
    model = Net_screen(train_dataset.num_features, 2, args).to(device)
elif args.model_type == 'Net_screen_DTI':
    model = Net_screen_DTI(train_dataset.num_features, 2, args).to(device)
elif args.model_type == 'Net_screen_energy':
    model = Net_screen_energy(train_dataset.num_features, 2, args).to(device)
elif args.model_type == 'Net_graph_DTI':
    model = Net_graph_DTI(train_dataset.num_features, args).to(device)

if args.pre_model != 'None':
    model = torch.load(args.pre_model).to(device)
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
if args.loss == 'CrossEntropyLoss':
    loss_op = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1.0,weight]).to(device))
if args.class_dir:
    # loss_op = loss_fn_dir(device)
    loss_op = torch.nn.CrossEntropyLoss()
    assert args.model_type == 'Net_coor_dir'
# loss_op_kld = torch.nn.KLDivLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train(epoch):
    model.train()

    total_loss = 0
    tot = 0
    positive = 0
    negative = 0
    t = time()
    pbar = tqdm(total=train_loader_size // args.batch_size + 1)
    pbar.set_description('Training poses...')
    for data in train_loader:
        data = data.to(device)

        rmsd = data.y.square().sum(dim=1)
        rmsd = global_mean_pool(rmsd, data.batch[data.flexible_idx.bool()]).sqrt()
        data_y = (rmsd < args.th).long()
        if epoch == 1:
            positive += torch.count_nonzero(data_y).item()
            negative += data_y.size()[0] - torch.count_nonzero(data_y).item()
        optimizer.zero_grad()
        if args.model_type == 'Net_screen':
            pred = model(data.x, data.edge_index, data.dist, data.flexible_idx.bool(), data.batch)
        if args.model_type == 'Net_screen_energy':
            pred = model(data.x, data.edge_index, data.dist, data.flexible_idx.bool(), data.batch, data.energy, data.tot_energy)
        elif args.model_type == 'Net_screen_DTI':
            idx = data.dist[:,1] == 0
            index1 = data.edge_index[:, idx]
            dist1 = data.dist[idx]
            pred = model(data.x, index1, data.edge_index, dist1, data.dist, data.flexible_idx.bool(), data.batch)
        elif args.model_type == 'Net_graph_DTI':
            idx = data.dist[:,1] == 0
            index1 = data.edge_index[:, idx]
            dist1 = data.dist[idx]
            pred = model(data.x, index1, data.edge_index, dist1, data.dist, data.flexible_idx.bool(), data.batch)
        loss = loss_op(pred, data_y)
        total_loss += loss.item() * args.batch_size
        loss.backward()
        optimizer.step()
        tot += 1
        # break
        pbar.update(1)
    pbar.close()
    if epoch == 1:
        print(f'positive: {positive}, negative: {negative}: weight: {negative / positive}')
    
    print(f"trained {tot} batches, take {time() - t}s")
    return total_loss / train_loader_size


@torch.no_grad()
def test(loader, epoch):
    model.eval()
    
    correct = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    tg = 0
    fg = 0
    total_loss = 0
    labels = []
    result = []
    labels_wogt = []
    result_wogt = []
    avg_selected_rmsd = []
    avg_unselected_rmsd = []
    pbar = tqdm(total=test_loader_size + 1)
    pbar.set_description('Testing poses...')
    for data in loader:
        data = data.to(device)
        rmsd = data.y.square().sum(dim=1)
        rmsd = global_mean_pool(rmsd, data.batch[data.flexible_idx.bool()]).sqrt()
        data_y = (rmsd < args.th).long()
        if args.model_type == 'Net_screen':
            # out = model(data.x.to(device), data.edge_index.to(device), data.dist.to(device), data.flexible_idx.bool().to(device), data.batch.to(device))
            out = model(data.x, data.edge_index, data.dist, data.flexible_idx.bool(), data.batch)
        if args.model_type == 'Net_screen_energy':
            out = model(data.x, data.edge_index, data.dist, data.flexible_idx.bool(), data.batch, data.energy, data.tot_energy)
        elif args.model_type == 'Net_screen_DTI':
            idx = data.dist[:,1] == 0
            index1 = data.edge_index[:, idx]
            if index1.size()[1] == 0:
                continue
            # continue
            dist1 = data.dist[idx]
            out = model(data.x, index1, data.edge_index, dist1, data.dist, data.flexible_idx.bool(), data.batch)

        elif args.model_type == 'Net_graph_DTI':
            idx = data.dist[:,1] == 0
            index1 = data.edge_index[:, idx]
            if index1.size()[1] == 0:
                continue
            dist1 = data.dist[idx]
            out = model(data.x, index1, data.edge_index, dist1, data.dist, data.flexible_idx.bool(), data.batch)


        loss = loss_op(out, data_y)

        total_loss += loss.item() * data.num_graphs
        result = result + out.cpu()[:,1].tolist()
        pred = out.max(1)[1]
        correct += pred.eq(data_y).cpu().sum().item()
        
        out=pred.cpu().numpy()
        label=data_y.cpu().numpy()
        rmsd=rmsd.cpu().numpy()
        for i in range(out.shape[0]):
            if out[i] == 0 and label[i] == 0:
                tn += 1
                avg_unselected_rmsd.append(rmsd[i])
            if out[i] == 0 and label[i] == 1:
                fn += 1
                avg_unselected_rmsd.append(rmsd[i])
            if out[i] == 1 and label[i] == 0:
                fp += 1
                avg_selected_rmsd.append(rmsd[i])
            if out[i] == 1 and label[i] == 1:
                tp += 1
                avg_selected_rmsd.append(rmsd[i])
        labels = labels + data_y.cpu().tolist()
        assert len(label) == data_y.size()[0]
        pbar.update(1)
    pbar.close()

    fpr, tpr, thresholds = metrics.roc_curve(labels, result, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    avg_selected_rmsd_ = -1
    avg_unselected_rmsd_ = -1
    if fp + tp > 0:
        avg_selected_rmsd_ = sum(avg_selected_rmsd) / (fp + tp) * 100.0
    if fn + tn > 0:
        avg_unselected_rmsd_ = sum(avg_unselected_rmsd) / (fn + tn) * 100.0

    plot.rmsd_hist(np.array(avg_selected_rmsd)*100,
                   os.path.join(args.plt_dir, f"rmsd_hist_final"),
                   xlim=20,
                   title='Poses Selected by Pose-selection Model')
    plot.rmsd_hist(np.array(avg_unselected_rmsd)*100,
                   os.path.join(args.plt_dir, f"rmsd_hist_unselect_final"),
                   xlim=20,
                   title='Poses Selected by Pose-selection Model')
    print(f"tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}, auc: {auc}, acc: {(tp+tn) / (tp+tn+fp+fn)}, avg_rmsd: {avg_selected_rmsd_}, avg_rmsd_un: {avg_unselected_rmsd_}")
    strr = "Epoch: {:02d}, tp: {:.4f}, fp: {:.4f}, tn: {:.4f}, fn: {:.4f}, auc: {:.4f}, acc: {:.4f}, avg_rmsd: {:.6f}, avg_rmsd_un: {:.6f}".format(
        epoch, tp, fp, tn, fn, auc, (tp+tn) / (tp+tn+fp+fn), avg_selected_rmsd_, avg_unselected_rmsd_)
    with open(args.output+'/output', "a") as f:
        f.write(strr+'\n')


    # return correct / test_dataset_size, total_loss / test_loader_size, auc
    return (tp+tn) / (tp+tn+fp+fn), total_loss / (tp+tn+fp+fn), auc


if not os.path.isdir(args.output):
    os.makedirs(args.output)
if not os.path.isdir(args.model_dir):
    os.makedirs(args.model_dir)
if not os.path.isdir(args.plt_dir):
    os.makedirs(args.plt_dir)
if not args.test_mode:
    max_auc = 0.0
    best_epoch = 0
    min_loss = 9999.99
    for epoch in range(args.start_epoch, args.start_epoch + args.epoch):
        loss = train(epoch)
        print(f"Train Loss: {loss}")
        acc, loss, auc = test(test_loader, epoch)
        print(f"Epoch: {epoch} Test Loss: {loss}  acc: {acc}  auc: {auc}")

        if args.output != 'none':
            with open(args.output + '/out', 'a') as f:
                f.write(f"Epoch: {epoch} Test Loss: {loss}  acc: {acc}  auc: {auc}\n")
        # if epoch > 3 and max_auc < auc:
        if epoch > 3 and loss < min_loss:
            torch.save(model, os.path.join(args.model_dir, f'model_{epoch}.pt'))
            print(f"save model at epoch {epoch}, auc of {auc} !!!!!!!!")
            if args.output != 'none':
                with open(args.output + '/out', 'a') as f:
                    f.write(f"save model at epoch {epoch}, auc of {auc} loss {loss} !!!!!!!!\n")
            max_auc = auc
            min_loss = loss
            best_epoch = epoch
        print("")

    print(f"\nBest model at epoch {best_epoch}, auc is {max_auc}")
else:
    assert args.pre_model != 'None'
    epoch = 1
    acc, loss, auc = test(test_loader, epoch)
    print(f"Epoch: {epoch} Test Loss: {loss}  acc: {acc}  auc: {auc}")