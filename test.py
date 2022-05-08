import os
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.metrics import f1_score


from dataset import PDBBindCoorTest, PDBBindNextStep, PDBBindNextStep2, PDBBindCoor, PDBBindScreen, PDBBindScreen2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="path to the dataset", type=str, default='data/pdbbind/pdbbind_rmsd_srand_mv_1')
parser.add_argument("--pre_root", help="path to the dataset", type=str, default='data/pdbbind/pdbbind_rmsd_srand202/')
parser.add_argument("--model_dir", help="path to the dataset", type=str, default='pdb-gnn/plt_1_512_1_1/model_8.pt')
parser.add_argument("--dataset", help="which dataset", type=str, default='PDBBindCoorTest')
parser.add_argument("--gpu_id", help="which gpu to use", type=int, default=0)
args = parser.parse_args()
print(args)


path = args.path
if not os.path.isdir(path):
    os.makedirs(path)
gpu_id = args.gpu_id
# dataset = PDBBind(root=path)
#train_dataset = PDBBindCoor(root=path, split='train')
#test_dataset = PDBBindCoor(root=path, split='test')
if args.dataset == 'PDBBindCoorTest':
	test_dataset = PDBBindCoorTest(root=path, model_dir=args.model_dir, gpu_id=gpu_id, pre_root=args.pre_root, split='test')
	train_dataset = PDBBindCoorTest(root=path, model_dir=args.model_dir, gpu_id=gpu_id, pre_root=args.pre_root, split='train')
elif args.dataset == 'PDBBindNextStep':
	test_dataset = PDBBindNextStep(root=path, model_dir=args.model_dir, gpu_id=gpu_id, pre_root=args.pre_root, split='test')
	train_dataset = PDBBindNextStep(root=path, model_dir=args.model_dir, gpu_id=gpu_id, pre_root=args.pre_root, split='train')
elif args.dataset == 'PDBBindNextStep2':
	test_dataset = PDBBindNextStep2(root=path, model_dir=args.model_dir, gpu_id=gpu_id, pre_root=args.pre_root, split='test')
	train_dataset = PDBBindNextStep2(root=path, model_dir=args.model_dir, gpu_id=gpu_id, pre_root=args.pre_root, split='train')
elif args.dataset == 'PDBBindNextStep2_autodock':
	test_dataset = PDBBindNextStep2(root=path, model_dir=args.model_dir, gpu_id=gpu_id, pre_root=args.pre_root, split='test')
elif args.dataset =='PDBBindCoor':
	train_dataset = PDBBindCoor(root=path, split='train')
	test_dataset = PDBBindCoor(root=path, split='test')
elif args.dataset =='PDBBindCoor_autodock':
	test_dataset = PDBBindCoor(root=path, split='test', data_type = 'autodock')
elif args.dataset == 'PDBBindScreen':
	test_dataset = PDBBindScreen(root=path, model_dir=args.model_dir, gpu_id=gpu_id, split='test')
	train_dataset = PDBBindScreen(root=path, model_dir=args.model_dir, gpu_id=gpu_id, split='train')
elif args.dataset == 'PDBBindScreen2':
	test_dataset = PDBBindScreen2(root=path, model_dir=args.model_dir, gpu_id=gpu_id, split='test')
	train_dataset = PDBBindScreen2(root=path, model_dir=args.model_dir, gpu_id=gpu_id, split='train')
elif args.dataset == 'PDBBindMUV':
	test_dataset = PDBBindScreen(root=path, model_dir=args.model_dir, gpu_id=gpu_id, split='test', data_type='muv')
elif args.dataset == 'PDBBindMUV2':
	test_dataset = PDBBindScreen2(root=path, model_dir=args.model_dir, gpu_id=gpu_id, split='test', data_type='muv')
#import pdb; pdb.set_trace()
