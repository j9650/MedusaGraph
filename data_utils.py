import numpy as np
import math
import random
import os
import networkx as nx
from networkx.readwrite import json_graph
import json
from scipy.spatial import distance

SPACE = 100.0
COV_BOND_TH = 2.5

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


def centre_of_pocket(ligand):
    x = sum(line[1] for line in ligand) / len(ligand)
    y = sum(line[2] for line in ligand) / len(ligand)
    z = sum(line[3] for line in ligand) / len(ligand)
    
    return x, y, z


def file_to_gt_pose(groundtruth_dir, groundtruth_suffix, pdb, Atoms, Bonds, pocket_th):
    """add ground truth

    """
    protein_gt = []
    ligand_gt = []
    edge_gt = set()
    pocket_idx = []

    f = open(groundtruth_dir+'/'+pdb+'/'+pdb+groundtruth_suffix[1], 'r')
    tt=0
    atoms_num = 0
    atoms_idx = []
    for st in f:
        if (st[:13] == '@<TRIPOS>ATOM'):
            tt = 1
            atoms_num = 0
            atoms_idx = []
            continue
        if (st[:13] == '@<TRIPOS>BOND'):
            tt = 2
            continue
        if (st[:13] == '@<TRIPOS>SUBS'):
            tt = 0
            continue
        if (tt == 1):
            name, x, y, z, atom = line_to_coor(st, 'ligand_mol2')
            atoms_num += 1
            # if (atom != 'H') or (name in H_type):
            if (atom != 'H'):
                ligand_gt.append((name, x, y, z, atom))
                atoms_idx.append(atoms_num)
                if len(atoms_idx) >= 2 and atoms_idx[-1] != atoms_idx[-2] + 1:
                    print("there is a H between heavy atoms.")
            #print(st)
        if (tt == 2):
            ss = st.split()
            
            # x = int(ss[1]) - 1
            # y = int(ss[2]) - 1
            x = atoms_idx.index(int(ss[1])) if int(ss[1]) in atoms_idx else -1
            y = atoms_idx.index(int(ss[2])) if int(ss[2]) in atoms_idx else -1

            if x == -1 or y == -1:
                continue

            # bond = ss[3]
            # if (x <= len(ligand_gt) and y <= len(ligand_gt) and bond in Bonds):
            # edge_gt[(x, y)] = 1
            # edge_gt[(y, x)] = 1
            edge_gt.add((x, y))
            edge_gt.add((y, x))
            #print(st)
    f.close()

    cx, cy, cz = centre_of_pocket(ligand_gt)

    f = open(groundtruth_dir+'/'+pdb+'/'+pdb+groundtruth_suffix[0], 'r')
    for st in f:
        ss = st.split()
        if (ss[0] == 'ATOM'): # or (ss[0] == 'HETATM'):
            name, x, y, z, atom, idx = line_to_coor(st, 'protein_atom')
            if (atom != 'H'):
                protein_gt.append((name, x, y, z, atom, idx))
                if name == 'CA' and distance.euclidean([x, y, z], [cx, cy, cz]) < pocket_th:
                    pocket_idx.append(idx)
    f.close()
    protein_gt = [line for line in protein_gt if line[5] in pocket_idx]

    # print(len(protein_gt))
    # print(pocket_idx)
    # print([line[-1] for line in protein_gt])
    gt_pose = gen_3D_2_gt_pose(protein_gt, ligand_gt, Atoms, file_dir=None, use_protein=False)
    return gt_pose, protein_gt, ligand_gt, edge_gt


def gen_3D_2_gt_pose(protein, ligand, Atoms, file_dir, use_protein = True):
    """ Generate the 3d-coordinate of a ground truth pose with pdb format. 
    """

    gt_pose = []
    for line in ligand:
        x = line[1]
        y = line[2]
        z = line[3]
        gt_pose.append([x / SPACE, y / SPACE, z / SPACE])

    if use_protein:
        for line in protein:
            x = line[1]
            y = line[2]
            z = line[3]
            gt_pose.append([x / SPACE, y / SPACE, z / SPACE])

    return gt_pose


def gen_3D_2_pose(protein, ligand, Atoms, Bonds, bond_th, file_dir):
    """ Convert a pose with pdb format to graph format. 
    """
    
    nodes = []
    node_index = []
    edges = []
    dist = []
    
    feats = []
    node_id = 0

    for line in ligand:
        x = line[1]
        y = line[2]
        z = line[3]
        atom = line[4]

        nodes.append(node_id)
        feat = np.zeros(3 + len(Atoms))
        feat[Atoms.index(atom)] = 1
        feat[-3:] = [x / SPACE, y / SPACE, z / SPACE]
        feats.append(feat)
        node_id += 1

    ligand_nodes = node_id

    for line in protein:
        x = line[1]
        y = line[2]
        z = line[3]
        atom = line[4]

        nodes.append(node_id)
        feat = np.zeros(3 + len(Atoms))
        feat[Atoms.index(atom)] = 1
        feat[-3:] = [x / SPACE, y / SPACE, z / SPACE]
        feats.append(feat)
        node_id += 1


    assert node_id == len(nodes)

    tot = 0
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            c_i = feats[i][-3:]
            c_j = feats[j][-3:]
            dis = distance.euclidean(c_i, c_j)
            dis = round(dis*100000) / 100000
            if dis * SPACE < bond_th:
                # G.add_edge(i, j)
                if i < ligand_nodes and j < ligand_nodes:
                    # if (i, j) in covlent:
                    edges.append(j)
                    tot += 1
                    dist.append([dis, 0.0, 0.0])
                    # TODO: covalent bond
                    # dist[-1][Bonds.index(covlent[(i, j)]) + 3] = 1
                elif i >= ligand_nodes and j >= ligand_nodes:
                    edges.append(j)
                    tot += 1
                    dist.append([0.0, 0.0, dis])
                else:
                    edges.append(j)
                    tot += 1
                    dist.append([0.0, dis, 0.0])
        node_index.append(tot)
    
    with open(file_dir+"_data-G.json", 'a') as f:
        json.dump(node_index, f)
        f.write('\n')
        json.dump(edges, f)
        f.write('\n')
        json.dump(dist, f)
        f.write('\n')

    with open(file_dir+"_data-feats", 'ab') as f:
        np.save(f, feats)

    return len(nodes)


def gen_3D_2_pose_atomwise(protein, ligand, Atoms, Bonds, edge_gt, bond_th, file_dir):
    """ Convert a pose with pdb format to graph format. 
    """
    
    nodes = []
    node_index = []
    edges = []
    dist = []
    
    feats = []
    node_id = 0
    la = len(Atoms)

    for line in ligand:
        x = line[1]
        y = line[2]
        z = line[3]
        atom = line[4]

        nodes.append(node_id)
        feat = np.zeros(3 + 2 * la)
        feat[Atoms.index(atom) + la] = 1
        feat[-3:] = [x / SPACE, y / SPACE, z / SPACE]
        feats.append(feat)
        node_id += 1

    ligand_nodes = node_id

    for line in protein:
        x = line[1]
        y = line[2]
        z = line[3]
        atom = line[4]

        nodes.append(node_id)
        feat = np.zeros(3 + 2 * la)
        feat[Atoms.index(atom)] = 1
        feat[-3:] = [x / SPACE, y / SPACE, z / SPACE]
        feats.append(feat)
        node_id += 1


    assert node_id == len(nodes)

    tot = 0
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            c_i = feats[i][-3:]
            c_j = feats[j][-3:]
            dis = distance.euclidean(c_i, c_j)
            dis = round(dis*100000) / 100000
            if dis * SPACE < bond_th:
                # G.add_edge(i, j)
                if i < ligand_nodes and j < ligand_nodes:
                    if (i, j) in edge_gt:
                        edges.append(j)
                        tot += 1
                        dist.append([dis, 0.0, 0.0])
                        # TODO: covalent bond
                        # dist[-1][Bonds.index(covlent[(i, j)]) + 3] = 1
                elif i >= ligand_nodes and j >= ligand_nodes:
                    if dis *SPACE < COV_BOND_TH:
                        edges.append(j)
                        tot += 1
                        dist.append([0.0, 0.0, dis])
                else:
                    edges.append(j)
                    tot += 1
                    dist.append([0.0, dis, 0.0])
        node_index.append(tot)
    
    with open(file_dir+"_data-G.json", 'a') as f:
        json.dump(node_index, f)
        f.write('\n')
        json.dump(edges, f)
        f.write('\n')
        json.dump(dist, f)
        f.write('\n')

    with open(file_dir+"_data-feats", 'ab') as f:
        np.save(f, feats)

    return len(nodes)


def get_bonds_from_mol2(file_name):
    # should give a mol2 file
    f = open(file_name, 'r')
    tt=0
    edge_gt = set()
    for st in f:
        if (st[:13] == '@<TRIPOS>ATOM'):
            tt = 1
            continue
        if (st[:13] == '@<TRIPOS>BOND'):
            tt = 2
            continue
        if (st[:13] == '@<TRIPOS>SUBS'):
            tt = 0
            continue
        if (tt == 1):
            continue
        if (tt == 2):
            ss = st.split()
            x = int(ss[1]) - 1
            y = int(ss[2]) - 1
            edge_gt.add((x, y))
            edge_gt.add((y, x))
    f.close()

    return edge_gt