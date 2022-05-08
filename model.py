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

def loss_fn_dir(device):
    def loss_fn(outputs, labels):

        KD_loss = nn.CrossEntropyLoss()(outputs[:, 0:2], labels[:, 0])
        KD_loss += nn.CrossEntropyLoss()(outputs[:, 2:4], labels[:, 1])
        KD_loss += nn.CrossEntropyLoss()(outputs[:, 4:6], labels[:, 2])
        return KD_loss * 0.333333333

    return loss_fn

def loss_fn_cos(device, reduction):
    def loss_fn(outputs, labels):
        if reduction == 'sum':
            loss = (0.999 * F.cosine_similarity(outputs, labels)).acos().sum() + 0.1 * nn.MSELoss(reduction=reduction)(outputs, labels)
        elif reduction == 'mean':
            loss = (0.999 * F.cosine_similarity(outputs, labels)).acos().mean() + 0.1 * nn.MSELoss(reduction=reduction)(outputs, labels)
        return loss

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


class Net_screen(torch.nn.Module):
    def __init__(self, num_features, num_classes, args):
        super(Net_screen, self).__init__()
        print('num_features: ', num_features)
        print('num_classes: ', num_classes)
        self.conv1 = TransformerConv(num_features, args.d_graph_layer, edge_dim = 3)
        # self.conv2 = GATConv(256, 256)
        self.convs = torch.nn.ModuleList([TransformerConv(args.d_graph_layer, args.d_graph_layer, edge_dim = 3) for _ in range(args.n_graph_layer)])

        self.lins = torch.nn.ModuleList([torch.nn.Linear(args.d_graph_layer, args.d_FC_layer) if i == 0 else
                                         torch.nn.Linear(args.d_FC_layer, args.d_FC_layer) for i in range(args.n_FC_layer)])
        self.lin3 = torch.nn.Linear(args.d_FC_layer, num_classes)
        self.relu = torch.nn.ReLU()
        self.Dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.last = args.last
        self.flexible = args.flexible

    def forward(self, x, edge_index, edge_attr, flexible_idx, batchs, energy=None):
        # print('x: ', x.size())
        # print('edge_attr: ', edge_attr.size())
        x = self.conv1(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.Dropout(x)

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.relu(x)
            x = self.Dropout(x)

        # x = global_mean_pool(x, batchs)
        if self.flexible:
            x = global_add_pool(x[flexible_idx], batchs[flexible_idx])
        else:
            x = global_mean_pool(x, batchs)

        # print(x.size())
        x = self.lins[0](x)
        x = self.relu(x)
        x = self.Dropout(x)
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


class Net_screen_energy(torch.nn.Module):
    def __init__(self, num_features, num_classes, args):
        super(Net_screen_energy, self).__init__()
        print('num_features: ', num_features)
        print('num_classes: ', num_classes)
        self.conv1 = TransformerConv(num_features, args.d_graph_layer, edge_dim = 3)
        # self.conv2 = GATConv(256, 256)
        self.convs = torch.nn.ModuleList([TransformerConv(args.d_graph_layer, args.d_graph_layer, edge_dim = 3) for _ in range(args.n_graph_layer)])

        self.lin1 = torch.nn.Linear(args.d_FC_layer + 2, args.d_FC_layer)
        self.lins = torch.nn.ModuleList([torch.nn.Linear(args.d_graph_layer, args.d_FC_layer) if i == 0 else
                                         torch.nn.Linear(args.d_FC_layer, args.d_FC_layer) for i in range(args.n_FC_layer)])
        self.lin3 = torch.nn.Linear(args.d_FC_layer, num_classes)
        self.relu = torch.nn.ReLU()
        self.Dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.last = args.last
        self.flexible = args.flexible

    def forward(self, x, edge_index, edge_attr, flexible_idx, batchs, energy1, energy2):
        # print('x: ', x.size())
        # print('edge_attr: ', edge_attr.size())
        x = self.conv1(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.Dropout(x)

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.relu(x)
            x = self.Dropout(x)

        # x = global_mean_pool(x, batchs)
        if self.flexible:
            x = global_add_pool(x[flexible_idx], batchs[flexible_idx])
        else:
            x = global_mean_pool(x, batchs)

        # print(x.size())
        x = self.lins[0](x)
        x = self.relu(x)
        y = self.Dropout(x)

        x = torch.cat((y, energy1.view(energy1.size()[0], 1), energy2.view(energy2.size()[0], 1)), dim = 1)
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


class Net_screen_DTI(torch.nn.Module):
    def __init__(self, num_features, num_classes, args):
        super(Net_screen_DTI, self).__init__()
        print('num_features: ', num_features)
        print('num_classes: ', num_classes)
        self.conv1 = TransformerConv(num_features, args.d_graph_layer, edge_dim = 3)
        ## self.conv1 = torch.nn.Linear(num_features, args.d_graph_layer, bias = False)

        # self.conv2 = GATConv(256, 256)
        self.convs = torch.nn.ModuleList([TransformerConv(args.d_graph_layer, args.d_graph_layer, edge_dim = 3) for _ in range(args.n_graph_layer)])

        self.lins = torch.nn.ModuleList([torch.nn.Linear(args.d_graph_layer, args.d_FC_layer) if i == 0 else
                                         torch.nn.Linear(args.d_FC_layer, args.d_FC_layer) for i in range(args.n_FC_layer)])
        self.lin3 = torch.nn.Linear(args.d_FC_layer, num_classes)
        self.relu = torch.nn.ReLU()
        self.Dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.last = args.last
        self.flexible = args.flexible

    def forward(self, x, edge_index1, edge_index2, edge_attr1, edge_attr2, flexible_idx, batchs, energy=None):
        # print('x: ', x.size())
        # print('edge_attr: ', edge_attr.size())

        x1 = self.relu(self.conv1(x, edge_index1, edge_attr1))
        x2 = self.relu(self.conv1(x, edge_index2, edge_attr2))
        x = x2 - x1
        x = self.Dropout(x)
        ## x = self.conv1(x)

        for i in range(len(self.convs)):
            x1 = self.relu(self.convs[i](x, edge_index1, edge_attr1))
            x2 = self.relu(self.convs[i](x, edge_index2, edge_attr2))
            x = x2 - x1
            x = self.Dropout(x)

        # x = global_mean_pool(x, batchs)
        if self.flexible:
            x = global_add_pool(x[flexible_idx], batchs[flexible_idx])
        else:
            x = global_mean_pool(x, batchs)

        # print(x.size())
        ## x = self.lins[0](x)
        ## x = self.relu(x)
        ## x = self.Dropout(x)
        for i in range(len(self.lins)):
            x = self.lins[i](x)
            x = self.relu(x)
            x = self.Dropout(x)
        x = self.lin3(x)
        if not hasattr(self, 'last') or self.last == 'log':
            return F.log_softmax(x, dim=1)
        elif not hasattr(self, 'last') or self.last == 'sigmoid':
            return torch.sigmoid(x).view(-1) 
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


class Net_coor(torch.nn.Module):
    def __init__(self, num_features, args):
        super(Net_coor, self).__init__()
        print("get the model of Net_coor")
        self.conv1 = TransformerConv(num_features, args.d_graph_layer, edge_dim = args.edge_dim)
        # self.conv1 = torch.nn.Linear(num_features, args.d_graph_layer)

        self.convs = torch.nn.ModuleList([TransformerConv(args.d_graph_layer, args.d_graph_layer, edge_dim = args.edge_dim) for _ in range(args.n_graph_layer)])
        # self.convs = torch.nn.ModuleList([torch.nn.Linear(args.d_graph_layer, args.d_graph_layer) for _ in range(args.n_graph_layer)])

        self.convl = TransformerConv(args.d_graph_layer, 3, edge_dim = args.edge_dim)
        # self.convl = torch.nn.Linear(args.d_graph_layer, 3)

        self.relu = torch.nn.ReLU()
        self.elu = torch.nn.ELU()
        self.gelu = torch.nn.GELU()
        self.Dropout = torch.nn.Dropout(p=args.dropout_rate)

        if args.residue:
            self.residue = True
        else:
            self.residue = False

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        # x = self.conv1(x)
        x = self.gelu(x)
        x = self.Dropout(x)

        for i in range(len(self.convs)):
            # identity = x.clone()
            if self.residue:
                identity = x
            x = self.convs[i](x, edge_index, edge_attr)
            # x = self.convs[i](x)
            x = self.gelu(x)
            if self.residue:
                x = x + identity
            # x = x + identity
            x = self.Dropout(x)

        x = self.convl(x, edge_index, edge_attr)
        # x = self.convl(x)
        return x


class Net_coor_dir(torch.nn.Module):
    def __init__(self, num_features, args):
        super(Net_coor_dir, self).__init__()
        print("get the model of Net_coor_dir")
        self.conv1 = TransformerConv(num_features, args.d_graph_layer, edge_dim = args.edge_dim)

        self.convs = torch.nn.ModuleList([TransformerConv(args.d_graph_layer, args.d_graph_layer, edge_dim = args.edge_dim) for _ in range(args.n_graph_layer)])

        # self.convl = TransformerConv(, 3, edge_dim = args.edge_dim)
        if args.class_dir:
            self.linl = torch.nn.Linear(args.d_graph_layer, args.d_graph_layer)
            self.linl2 = torch.nn.Linear(args.d_graph_layer, 8)
        else:
            self.linl = torch.nn.Linear(args.d_graph_layer, args.d_graph_layer)
            self.linl2 = torch.nn.Linear(args.d_graph_layer, 3)
        self.class_dir = args.class_dir

        self.relu = torch.nn.ReLU()
        self.elu = torch.nn.ELU()
        self.gelu = torch.nn.GELU()
        self.Dropout = torch.nn.Dropout(p=args.dropout_rate)

        if args.residue:
            self.residue = True
        else:
            self.residue = False

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.gelu(x)
        x = self.Dropout(x)

        for i in range(len(self.convs)):
            if self.residue:
                identity = x
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.gelu(x)
            if self.residue:
                x = x + identity
            x = self.Dropout(x)

        # x = self.convl(x, edge_index, edge_attr)
        # x = self.gelu(x)
        x = self.linl(x)
        x = self.relu(x)
        x = self.linl2(x)
        if self.class_dir:
            x = F.log_softmax(x, dim=1)
        return x


class Net_coor_len(torch.nn.Module):
    def __init__(self, num_features, args):
        super(Net_coor_len, self).__init__()
        print("get the model of Net_coor_dir")
        self.conv1 = TransformerConv(num_features, args.d_graph_layer, edge_dim = args.edge_dim)

        self.convs = torch.nn.ModuleList([TransformerConv(args.d_graph_layer, args.d_graph_layer, edge_dim = args.edge_dim) for _ in range(args.n_graph_layer)])

        # self.convl = TransformerConv(, 3, edge_dim = args.edge_dim)
        self.linl = torch.nn.Linear(args.d_graph_layer, args.d_graph_layer)
        self.linl2 = torch.nn.Linear(args.d_graph_layer, 1)

        self.relu = torch.nn.ReLU()
        self.elu = torch.nn.ELU()
        self.gelu = torch.nn.GELU()
        self.Dropout = torch.nn.Dropout(p=args.dropout_rate)

        if args.residue:
            self.residue = True
        else:
            self.residue = False

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.gelu(x)
        x = self.Dropout(x)

        for i in range(len(self.convs)):
            if self.residue:
                identity = x
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.gelu(x)
            if self.residue:
                x = x + identity
            x = self.Dropout(x)

        x = self.linl(x)
        x = self.relu(x)
        x = self.linl2(x)
        return x


class Net_coor_cent(torch.nn.Module):
    def __init__(self, num_features, args):
        super(Net_coor_cent, self).__init__()
        print("get the model of Net_coor_dir")
        self.conv1 = TransformerConv(num_features, args.d_graph_layer, edge_dim = args.edge_dim)

        self.convs = torch.nn.ModuleList([TransformerConv(args.d_graph_layer, args.d_graph_layer, edge_dim = args.edge_dim) for _ in range(args.n_graph_layer)])

        # self.convl = TransformerConv(, 3, edge_dim = args.edge_dim)
        self.linl = torch.nn.Linear(args.d_graph_layer, args.d_graph_layer)
        self.linl2 = torch.nn.Linear(args.d_graph_layer, 3)

        self.relu = torch.nn.ReLU()
        self.elu = torch.nn.ELU()
        self.gelu = torch.nn.GELU()
        self.Dropout = torch.nn.Dropout(p=args.dropout_rate)

        if args.residue:
            self.residue = True
        else:
            self.residue = False

    def forward(self, x, edge_index, edge_attr, batchs, flexible_idx):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.gelu(x)
        x = self.Dropout(x)

        for i in range(len(self.convs)):
            if self.residue:
                identity = x
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.gelu(x)
            if self.residue:
                x = x + identity
            x = self.Dropout(x)

        x = global_mean_pool(x[flexible_idx], batchs[flexible_idx])

        x = self.linl(x)
        x = self.relu(x)
        x = self.linl2(x)
        return x


class Net_coor_res(torch.nn.Module):
    def __init__(self, num_features, args):
        super(Net_coor_res, self).__init__()
        print("get the model of Net_coor_res")
        self.heads = args.heads
        self.lin0 = torch.nn.Linear(num_features, args.d_graph_layer)
        # self.conv1 = TransformerConv(num_features, args.d_graph_layer, edge_dim = 3, heads = self.heads, beta=True)
        # self.ln1 = torch.nn.LayerNorm([args.d_graph_layer * self.heads])
        # self.conv1 = torch.nn.Linear(num_features, args.d_graph_layer)

        self.convs = torch.nn.ModuleList([TransformerConv(args.d_graph_layer, args.d_graph_layer // self.heads, edge_dim = args.edge_dim, heads = self.heads, beta=False) for i in range(args.n_graph_layer)])
        # self.convs = torch.nn.ModuleList([torch.nn.Linear(args.d_graph_layer, args.d_graph_layer) for _ in range(args.n_graph_layer)])
        self.lns = torch.nn.ModuleList([torch.nn.LayerNorm([args.d_graph_layer]) for i in range(args.n_graph_layer)])
        self.lns2 = torch.nn.ModuleList([torch.nn.LayerNorm([args.d_graph_layer]) for i in range(args.n_graph_layer)])
        # self.lns3 = torch.nn.ModuleList([torch.nn.LayerNorm([args.d_graph_layer]) for i in range(args.n_graph_layer)])

        # self.convl = TransformerConv(args.d_graph_layer, args.d_graph_layer // 4, edge_dim = 3, heads = self.heads, beta=True)
        # self.convl = torch.nn.Linear(args.d_graph_layer, 3)
        self.lnl = torch.nn.LayerNorm([args.d_graph_layer])

        
        if self.heads > 0:
            # self.lin1 = torch.nn.Linear(args.d_graph_layer * self.heads, args.d_graph_layer)
            self.lins = torch.nn.ModuleList([torch.nn.Linear(args.d_graph_layer, args.d_graph_layer) for i in range(args.n_graph_layer)])
            self.lins2 = torch.nn.ModuleList([torch.nn.Linear(args.d_graph_layer, args.d_graph_layer) for i in range(args.n_graph_layer)])
            self.linl = torch.nn.Linear(args.d_graph_layer, 3)
        

        self.relu = torch.nn.ReLU()
        self.elu = torch.nn.ELU()
        self.gelu = torch.nn.GELU()
        self.Dropout = torch.nn.Dropout(p=args.dropout_rate)

    def forward(self, x, edge_index, edge_attr):
        x = self.elu(self.lin0(x))

        for i in range(len(self.convs)):
            # identity = x.clone()
            identity = x
            x = self.lns[i](x)
            x = self.convs[i](x, edge_index, edge_attr)
            # x = self.gelu(x)
            # x = self.Dropout(x)
            if self.heads > 0:
            #     x = self.lns2[i](x)
                x = self.gelu(self.lins[i](x))
            #     x = self.Dropout(x)
            # x = torch.cat((identity, x), 1)
            identity = x + identity

            x = self.lns2[i](identity)
            x = self.elu(self.lins2[i](x))
            # x = self.convs[i](x)
            # x = x + self.gelu(self.lins2[i](identity))
            x = x + identity
            x = self.Dropout(x)

        # x = self.convl(x, edge_index, edge_attr)
        x = self.lnl(x)
        # if self.heads > 1:
        #     x = self.linl(x)
        x = self.linl(x)
        return x
