import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv, global_mean_pool, global_add_pool
import numpy as np


def get_soft_label(rmsd, args):
    if args.KD_soft == 'exp':
        l=np.log(0.5)/args.th
        soft_label = torch.exp(rmsd * l).reshape(rmsd.size()[0], 1)
        soft_label = torch.cat([1-soft_label, soft_label], -1)
    if args.KD_soft == 'ln':
        soft_label = torch.log((rmsd+0.0001) / args.th).reshape(rmsd.size()[0], 1)
        soft_label = torch.cat([soft_label, -soft_label], -1)
    return soft_label

def loss_fn_kd(weight, device):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = 0.5
    T = 1.0
    def loss_fn(outputs, labels, teacher_outputs):
        # print(teacher_outputs.size())
        weights = torch.tensor([[weight, weight] if i == 1 else [1.0, 1.0] for i in labels]).to(device)
        # print(outputs.size())
        # print(teacher_outputs.size())
        KD_loss = nn.KLDivLoss(reduction='none')(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
        ## KD_loss = nn.KLDivLoss(reduction='none')(outputs/T, teacher_outputs/T) * (alpha * T * T)
        # print(KD_loss.size())
        KD_loss = KD_loss * weights
        KD_loss = KD_loss.sum() / weights.sum()
        # KD_loss = nn.KLDivLoss()(outputs/T,
                                 # teacher_outputs/T) * (alpha * T * T) + nn.CrossEntropyLoss(weight=torch.Tensor([1.0,weight]).to(device))(outputs, labels) * (1. - alpha)

        KD_loss += nn.CrossEntropyLoss(weight=torch.Tensor([1.0,weight]).to(device))(outputs, labels) * (1. - alpha)
        return KD_loss

    return loss_fn


class Net(torch.nn.Module):
    def __init__(self, num_features, num_classes, args):
        super(Net, self).__init__()
        self.conv1 = GATConv(num_features, args.d_graph_layer)
        # self.conv2 = GATConv(256, 256)
        self.convs = torch.nn.ModuleList([GATConv(args.d_graph_layer, args.d_graph_layer) for _ in range(args.n_graph_layer)])

        self.lin1 = torch.nn.Linear(args.d_graph_layer, args.d_FC_layer)
        self.lins = torch.nn.ModuleList([torch.nn.Linear(args.d_FC_layer, args.d_FC_layer) for _ in range(args.n_FC_layer)])
        self.lin3 = torch.nn.Linear(args.d_FC_layer, num_classes)
        self.relu = torch.nn.ReLU()
        self.Dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.last = args.last

    def forward(self, x, edge_index, batchs, energy=None):
        # x = F.elu(self.conv1(x, edge_index))
        # if energy is not None:
        # s    print(energy.size())
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.Dropout(x)
        # x = F.elu(self.conv2(x, edge_index))
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = self.relu(x)
            x = self.Dropout(x)
        # x = self.conv2(x, edge_index)
        # x = self.relu(x)
        x = global_mean_pool(x, batchs)
        # print(x.size())
        x = self.lin1(y)
        x = self.relu(x)
        x = self.Dropout(x)
        
        for i in range(len(self.lins)):
            x = self.lins[i](x)
            x = self.relu(x)
            x = self.Dropout(x)
        x = self.lin3(x)
        if not hasattr(self, 'last') or self.last == 'log':
            return F.log_softmax(x, dim=1)
        elif not hasattr(self, 'last') or self.last == 'sigmoid':
            return torch.sigmoid(x)
        elif not hasattr(self, 'last') or self.last == 'logsigmoid':
            return F.logsigmoid(x)
        else:
            return F.softmax(x, dim=1)


class Net_en(torch.nn.Module):
    def __init__(self, num_features, num_classes, args):
        super(Net_en, self).__init__()
        self.conv1 = GATConv(num_features, args.d_graph_layer)
        # self.conv2 = GATConv(256, 256)
        self.convs = torch.nn.ModuleList([GATConv(args.d_graph_layer, args.d_graph_layer) for _ in range(args.n_graph_layer)])

        self.lin1 = torch.nn.Linear(args.d_FC_layer + 1, args.d_FC_layer)
        self.lins = torch.nn.ModuleList([torch.nn.Linear(args.d_graph_layer, args.d_FC_layer) if i == 0 else
                                         torch.nn.Linear(args.d_FC_layer, args.d_FC_layer) for i in range(args.n_FC_layer)])
        self.lin3 = torch.nn.Linear(args.d_FC_layer, num_classes)
        self.relu = torch.nn.ReLU()
        self.Dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.last = args.last

    def forward(self, x, edge_index, batchs, energy=None):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.Dropout(x)

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = self.relu(x)
            x = self.Dropout(x)

        # x = global_mean_pool(x, batchs)
        x = global_add_pool(x, batchs)

        # print(x.size())
        x = self.lins[0](x)
        x = self.relu(x)
        y = self.Dropout(x)

        x = torch.cat((y, energy), dim = 1)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.Dropout(x)
        x = x + y 
        for i in range(1, len(self.lins)):
            x = self.lins[i](x)
            x = self.relu(x)
            x = self.Dropout(x)
        x = self.lin3(x)
        if not hasattr(self, 'last') or self.last == 'log':
            return F.log_softmax(x, dim=1)
        elif not hasattr(self, 'last') or self.last == 'sigmoid':
            return torch.sigmoid(x)
        elif not hasattr(self, 'last') or self.last == 'logsigmoid':
            return F.logsigmoid(x)
        else:
            return F.softmax(x, dim=1)


class Net_en_trans(torch.nn.Module):
    def __init__(self, num_features, num_classes, args):
        super(Net_en_trans, self).__init__()
        self.conv1 = TransformerConv(num_features, args.d_graph_layer, edge_dim = 1)
        # self.conv2 = GATConv(256, 256)
        self.convs = torch.nn.ModuleList([TransformerConv(args.d_graph_layer, args.d_graph_layer, edge_dim = 1) for _ in range(args.n_graph_layer)])

        self.lin1 = torch.nn.Linear(args.d_FC_layer + 1, args.d_FC_layer)
        self.lins = torch.nn.ModuleList([torch.nn.Linear(args.d_graph_layer, args.d_FC_layer) if i == 0 else
                                         torch.nn.Linear(args.d_FC_layer, args.d_FC_layer) for i in range(args.n_FC_layer)])
        self.lin3 = torch.nn.Linear(args.d_FC_layer, num_classes)
        self.relu = torch.nn.ReLU()
        self.Dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.last = args.last

    def forward(self, x, edge_index, edge_attr, batchs, energy=None):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.Dropout(x)

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.relu(x)
            x = self.Dropout(x)

        # x = global_mean_pool(x, batchs)
        x = global_add_pool(x, batchs)

        # print(x.size())
        x = self.lins[0](x)
        x = self.relu(x)
        y = self.Dropout(x)

        x = torch.cat((y, energy), dim = 1)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.Dropout(x)
        x = x + y 
        for i in range(1, len(self.lins)):
            x = self.lins[i](x)
            x = self.relu(x)
            x = self.Dropout(x)
        x = self.lin3(x)
        if not hasattr(self, 'last') or self.last == 'log':
            return F.log_softmax(x, dim=1)
        elif not hasattr(self, 'last') or self.last == 'sigmoid':
            return torch.sigmoid(x)
        elif not hasattr(self, 'last') or self.last == 'logsigmoid':
            return F.logsigmoid(x)
        else:
            return F.softmax(x, dim=1)


class Net_en_rmsd(torch.nn.Module):
    def __init__(self, num_features, args):
        super(Net_en_rmsd, self).__init__()
        self.conv1 = GATConv(num_features, args.d_graph_layer)
        # self.conv2 = GATConv(256, 256)
        self.convs = torch.nn.ModuleList([GATConv(args.d_graph_layer, args.d_graph_layer) for _ in range(args.n_graph_layer)])

        self.lin1 = torch.nn.Linear(args.d_FC_layer + 1, args.d_FC_layer)
        self.lins = torch.nn.ModuleList([torch.nn.Linear(args.d_graph_layer, args.d_FC_layer) if i == 0 else
                                         torch.nn.Linear(args.d_FC_layer, args.d_FC_layer) for i in range(args.n_FC_layer)])
        self.lin3 = torch.nn.Linear(args.d_FC_layer, 1)
        self.relu = torch.nn.ReLU()
        self.Dropout = torch.nn.Dropout(p=args.dropout_rate)

    def forward(self, x, edge_index, batchs, energy=None):
        # x = F.elu(self.conv1(x, edge_index))
        # if energy is not None:
        # s    print(energy.size())
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.Dropout(x)
        # x = F.elu(self.conv2(x, edge_index))
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = self.relu(x)
            x = self.Dropout(x)
        # x = self.conv2(x, edge_index)
        # x = self.relu(x)
        
        # x = global_mean_pool(x, batchs)
        x = global_add_pool(x, batchs)
        # print(x.size())
        x = self.lins[0](x)
        x = self.relu(x)
        y = self.Dropout(x)

        x = torch.cat((y, energy), dim = 1)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.Dropout(x)
        x = x + y

        for i in range(1, len(self.lins)):
            x = self.lins[i](x)
            x = self.relu(x)
            x = self.Dropout(x)
        x = self.lin3(x)
        if not hasattr(self, 'last') or self.last == 'log':
            return F.log_softmax(x, dim=1)
        elif not hasattr(self, 'last') or self.last == 'sigmoid':
            return torch.sigmoid(x)
        elif not hasattr(self, 'last') or self.last == 'logsigmoid':
            return F.logsigmoid(x)
        else:
            return F.softmax(x, dim=1)