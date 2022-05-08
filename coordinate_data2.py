import numpy as np
import math
import random
import os
import networkx as nx
from networkx.readwrite import json_graph
import json
from scipy.spatial import distance
from data_utils import line_to_coor, gen_3D_2_pose_atomwise, gen_3D_2_gt_pose, file_to_gt_pose

from tqdm import tqdm

SPACE = 100.0
POCKET_TH = 12.0

def _write_gt_pose_to_file(file_dir, flexible_len, gt_pose, edge_gt, pdb):
    with open(file_dir+'_label', 'ab') as f:
        np.save(f, np.array([flexible_len]))
        np.save(f, np.array(gt_pose))
        np.save(f, np.array(list(edge_gt)))
        np.save(f, np.array(pdb))


def _pdb_file_to_pose(pdb, pdb_file, Atoms, Bonds, bond_th, protein_gt, tot, tile_size,
                      global_tot, output_dir, gt_pose, ligand_gt, edge_gt, file_counter):
    """
    generate poses
    """
    ligand = []
    min_energy = 0
    f = open(pdb_file,"r")
    flag_label = 0
    for st in f:
        ss = st.split()
        if (ss[0] == 'REMARK') and (ss[1] == 'POSE:'):
            continue
        if (ss[0] == 'REMARK') and (ss[1] == 'E_without_VDWR:'):
            this_pose_energy = float(ss[2])
            if this_pose_energy < min_energy:
                min_energy = this_pose_energy
        if (ss[0] == 'ENDMDL'):
            num_nodes = gen_3D_2_pose_atomwise(protein_gt, ligand, Atoms, Bonds, edge_gt, bond_th, output_dir+"/"+str(file_counter))

            if num_nodes != len(gt_pose) + len(protein_gt):
                print(f"pose has {num_nodes} nodes while gt has {len(gt_pose)} nodes. protein_gt {pdb_file[-8:-4]},  {len(ligand_gt)}")
                print(f"{len(protein_gt)} {len(ligand)} {len(gt_pose)}")
                print([line[-1] for line in protein_gt])

            _write_gt_pose_to_file(output_dir+"/"+str(file_counter), len(ligand), gt_pose, edge_gt, pdb)
            ligand = []
            tot = tot + 1
            global_tot = global_tot + 1

            if (tot >= tile_size):
                file_counter = file_counter + 1
                tot = 0
        if (ss[0] == 'HETATM'):
            name, x, y, z, atom = line_to_coor(st, 'ligand_pdb')
            if (atom != 'H'):
                ligand.append((name, x, y, z, atom))

    f.close()

    return tot, file_counter, global_tot


def _count_atoms_types(pdbbind_dir, rec_list, Atoms):

    """count Atoms types

    """
    poses_num = 0
    tot = len(rec_list)
    #tot = 0

    pbar = tqdm(total=len(rec_list))
    pbar.set_description('Counting Atoms type...')
    for line in rec_list:
        f = open(pdbbind_dir+'/'+line,'r')
        flag = 0
        for st in f:
            ss = st.split()
            #print(ss)
            if (ss[0] == 'REMARK') and (ss[1] == 'POSE:'):
                poses_num = poses_num + 1
                if flag == 1:
                    break
                flag = 1
            if (ss[0] == 'ATOM') or (ss[0] == 'HETATM'):
                #atom = ss[2][0]
                #if(atom >='0') and (atom <= '9'):
                #    atom = ss[2][1]
                atom = ss[-1]
                if (not atom in Atoms) and (atom != 'H'):
                # if not atom in Atoms:
                    Atoms.append(atom)
            if (ss[0] == 'ATOM') and (ss[2] == 'CA'):
                atom = ss[3]
                if (not atom in Atoms) and (atom != 'UNK'):
                    Atoms.append(atom)

        f.close()
        pbar.update(1)
            
    pbar.close()
    return Atoms


def read_pdbbind_to_disk_rmsd_energy_split(input_list,
                                           groundtruth_dir,
                                           groundtruth_suffix,
                                           pdbbind_dir,
                                           output_dir,
                                           resolution,
                                           tile_size,
                                           bond_th,
                                           pocket_th,
                                           pdb_id_st,
                                           pdb_id_ed,
                                           seed = None):
    # input_list: /gpfs/group/mtk2/cyberstar/hzj5142/GNN/GNN/DGNN/data/pdbbind/pdb_list_15844_
    #             /gpfs/group/mtk2/cyberstar/hzj5142/GNN/GNN/DGNN/data/pdbbind/pdb_list_
    # groundtruth_dir: /gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/pdb_v2018/data/
    #                  /gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/pdbbind/
    # pdbbind_dir: /gpfs/scratch/hzj5142/AtomNet/medusa/pdbbind_output_16126/s367/
    #              /gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/medusa/pdbbind_output_s367/
    rec_list = []
    label_list = []
    f_list = open(input_list, "r")
    for line in f_list:
        line_pdb = line.split()[0]
        if '.pdb' not in line_pdb:
            line_pdb = line_pdb + '.pdb'
        rec_list.append(line_pdb)
    f_list.close()

    Atoms = ['N', 'C', 'O', 'S', 'Br', 'Cl', 'P', 'F', 'I']
    Bonds = ['1', '2', 'ar', 'am']
    actives_n = []
    decoys_n = []
    actives_lists = []
    decoys_lists = []
    print("before Atom type count")

    # Atoms = _count_atoms_types(pdbbind_dir, rec_list, Atoms)

    print("total atoms: " + str(len(Atoms)))
    print(Atoms)


    """Generate 3D data files

    """

    G = nx.Graph()
    # tmp_feat = []

    tot = len(rec_list)
    tt = list(np.arange(tot))
    random_list = random.sample(tt, tot)

    tot = 0 # how many poses so far
    global_tot = 0
    file_counter = 0 # id of the file
    this_pose_energy = 0.0

    pdb_list = []
    f_list = open(input_list, "r")
    for line in f_list:
        pdb_list.append(line.strip())
    f_list.close()

    pbar = tqdm(total=pdb_id_ed - pdb_id_st)
    pbar.set_description('Generating poses...')
    for i in range(pdb_id_st, pdb_id_ed):

        pdb = pdb_list[i]
        gt_pose, protein_gt, ligand_gt, edge_gt = file_to_gt_pose(groundtruth_dir, groundtruth_suffix, pdb, Atoms, Bonds, pocket_th)
        pdb_file = os.path.join(pdbbind_dir, pdb+'.pdb')
        tot, file_counter, global_tot = _pdb_file_to_pose(pdb, pdb_file, Atoms, Bonds, bond_th, protein_gt, tot, tile_size,
                                                          global_tot, output_dir, gt_pose, ligand_gt, edge_gt, file_counter)
        pbar.update(1)
    pbar.close()


    print("3D data generated")
    print("total " + str(global_tot) + " poses generated")