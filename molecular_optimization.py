import os
import sys
import argparse
from time import time


'''
# obabel vina medusad
parser = argparse.ArgumentParser()
parser.add_argument("--input_list", help="list of train/test pdbs", type=str, default='/gpfs/group/mtk2/cyberstar/hzj5142/GNN/GNN/DGNN/data/pdbbind/pdb_list_test')
parser.add_argument("--pdbbind_dir", help="dir to pdbbind dataset", type=str, default='/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/pdbbind/')
parser.add_argument("--pdbbind_tmp_dir", help="dir to pdbbind dataset", type=str, default='/gpfs/scratch/hzj5142/GNN/GNN/DGNN/cv1/pdbbind_tmp')
parser.add_argument("--pdb_version", help="version of pdbbind", type=str, default = '2016')
parser.add_argument("--output", help="name of output file", type=str, default='medusa_auto_time_pdbbind')
parser.add_argument("--start_iter", help="test data from which pdb id", type=int, default = 0)
parser.add_argument("--step", help="how many pdb to be test", type=int, default = 751)
parser.add_argument("--obabel_binary", help="binary of obabel", type=str, default='obabel')
parser.add_argument("--vina_binary", help="binary of autodock vina", type=str, default='/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/autodock_vina_1_1_2_linux_x86/bin/vina')
parser.add_argument("--mgl_path", help="path to the autodock MGL tool", type=str, default='/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/MGLTool')
parser.add_argument("--medusa_binary", help="binary of medusadock", type=str, default='/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/medusa/medusa-nocu/bin/medusa')
parser.add_argument("--medusa_parameter", help="parameter of medusadock", type=str, default='/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/medusa/data/parameter')
args = parser.parse_args()
'''

'''
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", help="input mol2 file name", type=str, default='/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/medusa/data/parameter')
args = parser.parse_args()
'''

def line_to_coor(line, form):
    if form == 'protein':
        st = line.split()
        name = line[12:17].strip().strip('\'')
        x = float(line[30:38]) # - c_x
        y = float(line[38:46]) # - c_y
        z = float(line[46:54]) # - c_z
        atom = st[3]
        idx = int(line[22:26])
        return name, x, y, z, atom, idx
    if form == 'protein_atom':
        st = line.split()
        name = line[12:17].strip().strip('\'')
        x = float(line[30:38]) # - c_x
        y = float(line[38:46]) # - c_y
        z = float(line[46:54]) # - c_z
        atom = st[-1]
        idx = int(line[22:26])
        return name, x, y, z, atom, idx
    if form == 'ligand_pdb':
        st = line.split()
        name = line[12:17].strip().strip('\'')
        x = float(line[30:38]) # - c_x
        y = float(line[38:46]) # - c_y
        z = float(line[46:54]) # - c_z
        atom = st[-1]
    elif form == 'ligand_mol2':
        st = line.split()
        name = st[1].strip('\'')
        x = float(line[16:26])
        y = float(line[26:36])
        z = float(line[36:46])
        atom = st[5]
        atom = atom.split('.')[0]

    if len(name) > 3:
        name = name[:3]
    while not atom[0].isalpha():
        atom = atom[1:]
    while not atom[-1].isalpha():
        atom = atom[:-1]
    return name, x, y, z, atom


def _set_num(x, l):
	return ' ' * (l - len(str(x))) + str(x)


def _set_coord(x, l):
	xx = str(round(x, 4))
	return ' ' * (l - len(xx)) + xx


def get_refined_pose_file(mol2_file, output_mol2_file, ligand):

	lines = []
	flag = 0
	a2a = {}
	atom_id = 1
	nonH_atom = 1
	bond_id = 1
	with open(mol2_file, 'r') as f:
		for line in f:
			if (line[:13] == '@<TRIPOS>ATOM'):
				flag = 1
				lines.append(line)
				continue
			if (line[:13] == '@<TRIPOS>BOND'):
				flag = 2
				lines.append(line)
				assert len(ligand) == nonH_atom - 1
				continue
			if (line[:13] == '@<TRIPOS>SUBS'):
				flag = 0
				lines.append(line)
				continue
			if flag == 1:
				name, x, y, z, atom = line_to_coor(line, 'ligand_mol2')
				if (atom != 'H'):
					a2a[atom_id] = nonH_atom
					x = ligand[nonH_atom - 1][0]
					y = ligand[nonH_atom - 1][1]
					z = ligand[nonH_atom - 1][2]
					st = _set_num(nonH_atom, 7) + line[7:16] + _set_coord(x, 10) + _set_coord(y, 10) + _set_coord(z, 10) + line[46:]
					lines.append(st)
					nonH_atom += 1
				atom_id += 1
				continue
			if flag == 2:
				st = line.split()
				x = int(st[1])
				y = int(st[2])
				if x in a2a and y in a2a:
					x = a2a[x]
					y = a2a[y]
					l = _set_num(bond_id, 6) + _set_num(x, 5) + _set_num(y, 5) + line[16:]
					bond_id += 1
					lines.append(l)
				continue
			lines.append(line)
	# modify bond # and atom #
	for i in range(len(lines)):
		if '@<TRIPOS>MOLECULE' in lines[i]:
			l = i
			break
	# print(lines[l + 2])
	lines[l + 2] = _set_num(nonH_atom - 1, 5) + _set_num(bond_id - 1, 6) + lines[l + 2][11:]

	with open('tmp_mol.mol2', 'w') as f:
		for line in lines:
			f.write(line)

	os.system('obminimize tmp_mol.mol2 > tmp_mol.pdb')
	os.system(f'obabel -ipdb tmp_mol.pdb -omol2 -O {output_mol2_file}')
	os.system('rm tmp_mol.mol2')
	os.system('rm tmp_mol.pdb')






