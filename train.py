
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.metrics import f1_score
from sklearn import metrics

import sys
import os
import argparse
import numpy as np

from dataset import PDBBind
from model import Net, Net_en, Net_en_trans, loss_fn_kd, get_soft_label

parser = argparse.ArgumentParser()
parser.add_argument("--lr", help="learning rate", type=float, default = 0.0001)
parser.add_argument("--epoch", help="epoch", type=int, default = 1000)
parser.add_argument("--start_epoch", help="epoch", type=int, default = 1)
parser.add_argument("--batch_size", help="batch_size", type=int, default = 64)
parser.add_argument("--gpu_id", help="id of gpu", type=int, default = 3)
parser.add_argument("--data_path", help="train keys", type=str, default='/home/mdl/hzj5142/GNN/pdb-gnn/data/pdbbind_rmsd_srand4')
parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default = 1)
parser.add_argument("--d_graph_layer", help="number of GNN layer", type=int, default = 256)
parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default = 0)
parser.add_argument("--d_FC_layer", help="number of FC layer", type=int, default = 512)
parser.add_argument("--output", help="train result", type=str, default='output_train')
parser.add_argument("--model_dir", help="save best model", type=str, default='best_model.pt')
parser.add_argument("--pre_model", help="pre trained model", type=str, default='None')
parser.add_argument("--th", help="threshold for positive pose", type=float, default=3.00)
parser.add_argument("--dropout_rate", help="dropout rate", type=float, default=0.3)
parser.add_argument("--weight_bias", help="weight bias", type=float, default=1.0)
parser.add_argument("--last", help="activation of last layer", type=str, default='log')
parser.add_argument("--KD", help="if we apply knowledge distillation (Yes / No)", type=str, default='No')
parser.add_argument("--KD_soft", help="function convert rmsd to softlabel", type=str, default='exp')
parser.add_argument("--edge", help="if we use edge attr", type=str, default='False')
args = parser.parse_args()
print(args)


path = args.data_path

train_datasets = []
test_datasets = []
train_loaders = []
test_loaders = []

for dataset_name in os.listdir(path):
    train_datasets.append(PDBBind(root=os.path.join(path, dataset_name), split='train'))
    test_datasets.append(PDBBind(root=os.path.join(path, dataset_name), split='test'))
    train_loaders.append(DataLoader(train_datasets[-1], batch_size=args.batch_size, shuffle=True))
    test_loaders.append(DataLoader(test_datasets[-1], batch_size=args.batch_size))

train_loader_size = sum([len(loader.dataset) for loader in train_loaders])
test_dataset_size = sum([len(dataset) for dataset in test_datasets])
test_loader_size = sum([len(loader.dataset) for loader in test_loaders])

print(f"total {len(train_datasets)} subdatasets")
print(f"train_loader_size: {train_loader_size}")
print(f"test_dataset_size: {test_dataset_size}, test_loader_size: {test_loader_size}")


weight = 4.100135326385498 + args.weight_bias
print(f"weight: 1:{weight}")

gpu_id = str(args.gpu_id)
device_str = 'cuda:' + gpu_id if torch.cuda.is_available() else 'cpu'

device = torch.device(device_str)
print('cuda' if torch.cuda.is_available() else 'cpu')

if args.edge == 'True':
    print(args.edge)
    model = Net_en_trans(train_datasets[0].num_features, train_datasets[0].num_classes, args).to(device)
else:
    print(args.edge)
    model = Net_en(train_datasets[0].num_features, train_datasets[0].num_classes, args).to(device)

if args.pre_model != 'None':
    model = torch.load(args.pre_model).to(device)
if args.KD == 'No':
    loss_op = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1.0,weight]).to(device))
else:
    loss_op = loss_fn_kd(weight, device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train():
    model.train()

    total_loss = 0
    tot = 0
    for train_loader in train_loaders:
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            if args.edge == 'True':
                pred = model(data.x, data.edge_index, data.dist, data.batch, energy = data.energy.view(data.energy.size()[0], 1))
            else:
                pred = model(data.x, data.edge_index, data.batch, energy = data.energy.view(data.energy.size()[0], 1))
            if args.KD == 'No':
                loss = loss_op(pred, data.y)
            else:
                soft_label = get_soft_label(data.rmsd, args)
                loss = loss_op(pred, data.y, soft_label)
            total_loss += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
            tot += 1
    
    print(f"trained {tot} batches")
    return total_loss / train_loader_size


@torch.no_grad()
def test(loaders, epoch):
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
    for loader in loaders:
        for data in loader:
            if args.edge == 'True':
                out = model(data.x.to(device), data.edge_index.to(device), data.dist.to(device), data.batch.to(device), energy = data.energy.view(data.energy.size()[0], 1).to(device))
            else:
                out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device), energy = data.energy.view(data.energy.size()[0], 1).to(device))
            if args.KD == 'No':
                loss = loss_op(out, data.y.to(device))
            else:
                soft_label = get_soft_label(data.rmsd, args)
                loss = loss_op(out, data.y.to(device), soft_label.to(device))
            total_loss += loss.item() * data.num_graphs
            result = result + out.cpu()[:,1].tolist()
            result_wogt = result_wogt + [out.cpu().numpy()[i,1] for i in range(data.rmsd.size()[0]) if data.rmsd[i] != 0.0]
            pred = out.max(1)[1]
            correct += pred.cpu().eq(data.y).sum().item()
    
            out=pred.cpu().numpy()
            label=data.y.numpy()
            for i in range(out.shape[0]):
                if data.rmsd[i] == 0.0:
                    if out[i] == 0:
                        fg += 1
                    else:
                        tg += 1
                    continue
                if out[i] == 0 and label[i] == 0:
                    tn += 1
                if out[i] == 0 and label[i] == 1:
                    fn += 1
                if out[i] == 1 and label[i] == 0:
                    fp += 1
                if out[i] == 1 and label[i] == 1:
                    tp += 1
            labels = labels + data.y.tolist()
            assert len(label) == data.rmsd.size()[0]
            labels_wogt = labels_wogt + [label[i] for i in range(len(label)) if data.rmsd[i] != 0.0]

    fpr, tpr, thresholds = metrics.roc_curve(labels, result, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    fpr, tpr, thresholds = metrics.roc_curve(labels_wogt, result_wogt, pos_label=1)
    auc_wogt = metrics.auc(fpr, tpr)

    print(f"tg: {tg}, fg: {fg}, tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}, auc: {auc}")
    strr = "Epoch: {:02d}, tg: {:.4f}, fg: {:.4f}, tp: {:.4f}, fp: {:.4f}, tn: {:.4f}, fn: {:.4f}, auc: {:.4f}, acc: {:.4f}, auc: {:.4f}, acc: {:.4f}".format(
        epoch, tg, fg, tp, fp, tn, fn, auc, correct / test_dataset_size, auc_wogt, (tp+tn) / (tp+tn+fp+fn))
    with open(args.output+'put', "a") as f:
        f.write(strr+'\n')


    return correct / test_dataset_size, total_loss / test_loader_size, auc_wogt

best_auc = 0.0
best_epoch = 0
if not os.path.isdir(args.model_dir):
    os.makedirs(args.model_dir)
for epoch in range(args.start_epoch, args.start_epoch + args.epoch):
    loss = train()
    test_f1, loss_test, auc = test(test_loaders, epoch)
    strr = 'Epoch: {:02d}, Loss: {:.4f}, Test: {:.4f}, Test_loss: {:.4f}, auc: {:.4f}'.format(
        epoch, loss, test_f1, loss_test, auc)
    print(strr)
    with open(args.output, "a") as f:
        f.write(strr+'\n')

    if epoch > 5 and auc > best_auc:
        best_auc = auc
        best_epoch = epoch
        with open(args.output+'put', "a") as f:
            f.write(f"save at epoch {epoch}"+'\n')

    if epoch > 10:
        torch.save(model, os.path.join(args.model_dir, f"model_iter{epoch}.pt"))
