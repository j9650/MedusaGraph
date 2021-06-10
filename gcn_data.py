import numpy as np
import math
import random
import os
import networkx as nx
from networkx.readwrite import json_graph
import json
from scipy.spatial import distance

import dataset_from_file as dff

# def dist(x1, y1, z1, x2, y2, z2):
#     return sqrt

def gen_3D_2_split(protein, ligand, Atoms, space, resolution, file_dir):

    x=0.0
    y=0.0
    z=0.0

    for line in ligand:
        x = x + float(line[30:38])
        y = y + float(line[38:46])
        z = z + float(line[46:54])

    c_x = x / len(ligand)
    c_y = y / len(ligand)
    c_z = z / len(ligand)

    # G.clear()
    nodes = []
    node_index = []
    edges = []
    dist = []
    
    feats = []
    node_id = 0
    for line in protein:
        st = line.split()
        x = float(line[30:38]) # - c_x
        y = float(line[38:46]) # - c_y
        z = float(line[46:54]) # - c_z
        atom = 'p'+st[-1]
        if atom == 'pH':
            continue

        if distance.euclidean((x, y, z), (c_x, c_y, c_z)) <= space:
            # G.add_node(node_id)
            nodes.append(node_id)
            feat = np.zeros(3 + len(Atoms))
            feat[Atoms.index(atom)] = 1
            feat[-3:] = [(x - c_x) / space, (y - c_y) / space, (z - c_z) / space]
            feats.append(feat)
            node_id += 1

    for line in ligand:
        st = line.split()
        x = float(line[30:38]) # - c_x
        y = float(line[38:46]) # - c_y
        z = float(line[46:54]) # - c_z
        atom = 'l'+st[-1]

        if distance.euclidean((x, y, z), (c_x, c_y, c_z)) <= space:
            # G.add_node(node_id)
            nodes.append(node_id)
            feat = np.zeros(3 + len(Atoms))
            feat[Atoms.index(atom)] = 1
            feat[-3:] = [(x - c_x) / space, (y - c_y) / space, (z - c_z) / space]
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
            if dis * space < 6:
                # G.add_edge(i, j)
                edges.append(j)
                dist.append(dis)
                tot += 1
        node_index.append(tot)
    

    # feats = np.array(feats)

    # print(f'# of nodes: {len(G.nodes())}')
    with open(file_dir+"_data-G.json", 'a') as f:
        # json.dump(json_graph.node_link_data(G), f)
        json.dump(node_index, f)
        f.write('\n')
        json.dump(edges, f)
        f.write('\n')
        json.dump(dist, f)
        f.write('\n')

    with open(file_dir+"_data-feats", 'ab') as f:
        np.save(f, feats)

    # return feats


def write_data_to_file_easy_rmsd(file_dir, data_y):
    # with open(file_dir+"_data-G.json", 'w') as f:
    #     json.dump(data_g, f)
    # np.savez(file_dir+"_data-feats", *data_f)

    num = data_y.shape[0]
    f = open(file_dir+"_label", "a")
    for i in range(num):
        if (data_y[i, 0] == 1 and data_y[i, 1] == 0):
            f.write('0')
        else:
            f.write('1')
        if data_y.shape[1] == 3:
            f.write(' '+str(data_y[i,2])+'\n')
        elif data_y.shape[1] == 4:
            f.write(' '+str(data_y[i,2])+' '+str(data_y[i,3])+'\n')
    f.close()


def read_pdbbind_to_disk_rmsd_energy_split(input_list, label_list_file, pdbbind_dir, groundtruth_dir, output_dir, resolution, tile_size, seed = None):

    rec_list = []
    label_list = []
    f_list = open(input_list, "r")
    for line in f_list:
        line_pdb = line.split()[0]
        if '.pdb' not in line_pdb:
            line_pdb = line_pdb + '.pdb'
        rec_list.append(line_pdb)
        #label_list.append(line.split()[1])
    f_list.close()

    label_file_list = []
    label_file_list_rmsd = []
    f_list = open(label_list_file, "r")
    for line in f_list:
        rmsd = float(line.split()[0])
        label_file_list_rmsd.append(rmsd)
        if (rmsd < 3):
            label_file_list.append(1)
        else:
            label_file_list.append(0)
    f_list.close()

    #Atoms = []
    # Atoms = ['N', 'C', 'O', 'H', 'S', 'Br', 'Cl', 'P', 'F', 'I']
    Atoms = ['N', 'C', 'O', 'S', 'Br', 'Cl', 'P', 'F', 'I']
    actives_n = []
    decoys_n = []
    actives_lists = []
    decoys_lists = []
    print("before Atom type count")

    """count Atoms types

    """
    poses_num = 0
    tot = len(rec_list)
    #tot = 0
    for line in rec_list:
        #ligand_list = os.listdir(DUDE_dir+"/"+line+"/ac")
        f = open(pdbbind_dir+'/'+line,"r")
        #print("/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/medusa/output/"+line+".pdb")
        for st in f:
            ss = st.split()
            #print(ss)
            if (ss[0] == 'REMARK') and (ss[1] == 'POSE:'):
                poses_num = poses_num + 1
            if (ss[0] == 'ATOM') or (ss[0] == 'HETATM'):
                #atom = ss[2][0]
                #if(atom >='0') and (atom <= '9'):
                #    atom = ss[2][1]
                atom = ss[-1]
                if (not atom in Atoms) and (atom != 'H'):
                # if not atom in Atoms:
                    Atoms.append(atom)

        f.close()

        
    #neg_data = 3000
    print("total data: " + str(tot))
    print("total poses: " + str(poses_num))

    Atoms_new = []
    for atom in Atoms:
        Atoms_new.append('p'+atom)
        Atoms_new.append('l'+atom)
    Atoms = Atoms_new
    Atoms.append('lH')
    print("total atoms: " + str(len(Atoms)))
    print(Atoms)


    """Generate 3D data files

    """
    #tot = pos_data + neg_data
    kongge = '          '
    kongge = kongge + kongge
    kongge = kongge + kongge
    kongge = kongge + kongge


    # tmp_data = np.zeros((tile_size, resolution, resolution, resolution, len(Atoms)))
    # tmp_data = []
    G = nx.Graph()
    # tmp_feat = []
    tmp_label = np.zeros((tile_size, 4))

    tot = len(rec_list)
    tt = list(np.arange(tot))
    random_list = random.sample(tt, tot)

    tot = 0
    protein = []
    ligand = []
    cc = 0
    global_tot = 0
    file_counter = 0
    this_pose_energy = 0.0
    # for i in range(len(rec_list) // 2, len(rec_list)):
    for i in range(len(rec_list)):
        H_type = []

        line = rec_list[i]
        # print(line, ' start')

        min_energy = 0
        f = open(pdbbind_dir+"/"+line,"r")
        flag_label = 0
        for st in f:
            ss = st.split()
            if (ss[0] == 'REMARK') and (ss[1] == 'POSE:'):
                if(flag_label == 0):
                    print(i,' ',global_tot)
                    flag_label = 1
                global_tot = global_tot + 1
            # if (ss[0] == 'REMARK') and (ss[1] == 'E_total:'):
            if (ss[0] == 'REMARK') and (ss[1] == 'E_without_VDWR:'):
                this_pose_energy = float(ss[2])
                if this_pose_energy < min_energy:
                    min_energy = this_pose_energy
            if (ss[0] == 'ENDMDL'):
                # tmp_data[tot] = gen_3D_2_split(protein, ligand, Atoms, 20, resolution, tmp_data[tot])
                gen_3D_2_split(protein, ligand, Atoms, resolution, resolution, output_dir+"/"+str(file_counter))
                # tmp_data.append(G)
                # tmp_feat.append(feats)
                protein = []
                ligand = []
                if (label_file_list[cc] == 1):
                    tmp_label[tot, 0] = 0
                    tmp_label[tot, 1] = 1
                else:
                    tmp_label[tot, 0] = 1
                    tmp_label[tot, 1] = 0
                tmp_label[tot, 2] = label_file_list_rmsd[cc]
                tmp_label[tot, 3] = this_pose_energy
                tot = tot + 1
                cc = cc + 1

                if (tot >= tile_size):
                    write_data_to_file_easy_rmsd(output_dir+"/"+str(file_counter), tmp_label)
                    file_counter = file_counter + 1
                    tot = 0
                    # tmp_data.fill(0)
                    # tmp_data.clear()
                    # tmp_feat.clear()
                    tmp_label.fill(0)
                
            if (ss[0] == 'ATOM'):
                protein.append(st[:-1])
            if (ss[0] == 'HETATM'):
                ligand.append(st[:-1])
                if (ss[-1] == 'H') and (not ss[-1] in H_type):
                    H_type.append(ss[2])

        f.close()

        # if seed > 3:
        #     continue
        """add ground truth

        """
        protein = []
        ligand = []
        f = open(groundtruth_dir+'/'+line[:4]+'/'+line[:4]+'.rec.pdb', 'r')
        for st in f:
            ss = st.split()
            if (ss[0] == 'ATOM') or (ss[0] == 'HETATM'):
                protein.append(st[:-1])
        f.close()

        f = open(groundtruth_dir+'/'+line[:4]+'/'+line[:4]+'.lig.mol2', 'r')
        tt=0
        for st in f:
            #ss = st.split()
            if (st[:13] == '@<TRIPOS>ATOM'):
                tt = 1
                continue
            if (st[:13] == '@<TRIPOS>BOND'):
                tt = 0
                continue
            if (tt == 1):
                ss = kongge[:30]
                xx = str(round(float(st[16:26]), 3))
                ss = ss+kongge[:8-len(xx)]+xx
                xx = str(round(float(st[26:36]), 3))
                ss = ss+kongge[:8-len(xx)]+xx
                xx = str(round(float(st[36:46]), 3))
                ss = ss+kongge[:8-len(xx)]+xx
                xx = st.split()[5]
                xx = xx.split('.')[0]
                ss = ss + kongge[:10]+xx
                if (xx != 'H') or (st.split()[1] in H_type):
                    ligand.append(ss)
                #print(st)
        f.close()
        # tmp_data[tot] = gen_3D_2_split(protein, ligand, Atoms, 20, resolution, tmp_data[tot])
        gen_3D_2_split(protein, ligand, Atoms, resolution, resolution, output_dir+"/"+str(file_counter))
        # tmp_data.append(G)
        # tmp_feat.append(feats)
        tmp_label[tot, 0] = 0
        tmp_label[tot, 1] = 1
        tmp_label[tot, 2] = 0.0
        tmp_label[tot, 3] = min_energy - 1.0
        # tmp_label[tot, 3] = -41.7
        tot = tot + 1
        protein = []
        ligand = []
        if (tot >= tile_size):
            write_data_to_file_easy_rmsd(output_dir+"/"+str(file_counter), tmp_label)
            file_counter = file_counter + 1
            tot = 0
            # tmp_data.fill(0)
            # tmp_data.clear()
            # tmp_feat.clear()
            tmp_label.fill(0)
        


    if (tot>0):
        print('tot:', tot)
        write_data_to_file_easy_rmsd(output_dir+"/"+str(file_counter), tmp_label[:tot,:])
        file_counter = file_counter + 1
        tot = 0
        # tmp_data.fill(0)
        # tmp_data.clear()
        # tmp_feat.clear()
        tmp_label.fill(0)

    print("3D data generated")
