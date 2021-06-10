import numpy as np
import math
import random
import os

def read_data_from_file_easy_rmsd(file_dir, data_x, data_y):
    f = open(file_dir+"_data", "r")

    st = f.readline().split()
    num = int(st[0])
    x = int(st[1])
    y = int(st[2])
    z = int(st[3])
    c = int(st[4])

    if (data_x.shape[0]<num or data_x.shape[4]<c):
        data_x = np.zeros((num, x, y, z, c))
        data_y = np.zeros((num))

    data_x.fill(0)
    data_y.fill(0)
    for i in range(num):
        #if(i%10 == 0):
        #    print(i)
        tot = int(f.readline())
        for iiiii in range(tot):
            st = f.readline().split()
            ix = int(st[0])
            iy = int(st[1])
            iz = int(st[2])
            ic = int(st[3])
            value = math.trunc(float(st[4]))

            data_x[i, ix, iy, iz, ic] = value
    f.close()

    f = open(file_dir+"_label", "r")
    for i in range(num):
        st = f.readline().split()
        data_y[i] = float(st[1])
    f.close()

    return data_x, data_y, num

def read_data_from_file_withenergy_limitgt(file_dir, data_x, data_y, true_th = 3, false_th = 3, gt_tot = 0):
    f = open(file_dir+"_data", "r")

    st = f.readline().split()
    num = int(st[0])
    x = int(st[1])
    y = int(st[2])
    z = int(st[3])
    c = int(st[4])

    if (data_x.shape[0]<num or data_x.shape[4]<c):
        data_x = np.zeros((num, x, y, z, c))
        data_y = np.zeros((num, 4))

    cc = 0
    fl = open(file_dir+"_label", "r")
    data_x.fill(0)
    data_y.fill(0)
    for i in range(num):
        #if(i%10 == 0):
        #    print(i)
        st = fl.readline().split()
        rmsd = float(st[1])
        energy = float(st[2])
        if (rmsd == 0):
            if gt_tot < 3100:
                data_y[cc, 0] = 0
                data_y[cc, 1] = 1
                data_y[cc, 2] = rmsd
                data_y[cc, 3] = energy
                gt_tot += 1
        elif (rmsd <= true_th):
            data_y[cc, 0] = 0
            data_y[cc, 1] = 1
            data_y[cc, 2] = rmsd
            data_y[cc, 3] = energy
        elif (rmsd > false_th):
            data_y[cc, 0] = 1
            data_y[cc, 1] = 0
            data_y[cc, 2] = rmsd
            data_y[cc, 3] = energy
        else:
            tot = int(f.readline())
            for iiiii in range(tot):
                st = f.readline().split()
                #ix = int(st[0])
                #iy = int(st[1])
                #iz = int(st[2])
                #ic = int(st[3])
                #value = math.trunc(float(st[4]))

            continue

        tot = int(f.readline())
        for iiiii in range(tot):
            st = f.readline().split()
            ix = int(st[0])
            iy = int(st[1])
            iz = int(st[2])
            ic = int(st[3])
            value = math.trunc(float(st[4]))

            data_x[cc, ix, iy, iz, ic] = value
        cc = cc + 1
    f.close()
    fl.close()

    return data_x, data_y, cc, gt_tot


def read_data_from_file_withenergy(file_dir, data_x, data_y, true_th = 3, false_th = 3):
    f = open(file_dir+"_data", "r")

    st = f.readline().split()
    num = int(st[0])
    x = int(st[1])
    y = int(st[2])
    z = int(st[3])
    c = int(st[4])

    if (data_x.shape[0]<num or data_x.shape[4]<c):
        data_x = np.zeros((num, x, y, z, c))
        data_y = np.zeros((num, 4))

    cc = 0
    fl = open(file_dir+"_label", "r")
    data_x.fill(0)
    data_y.fill(0)
    for i in range(num):
        #if(i%10 == 0):
        #    print(i)
        st = fl.readline().split()
        rmsd = float(st[1])
        energy = float(st[2])
        if (rmsd <= true_th):
            data_y[cc, 0] = 0
            data_y[cc, 1] = 1
            data_y[cc, 2] = rmsd
            data_y[cc, 3] = energy
        elif (rmsd > false_th):
            data_y[cc, 0] = 1
            data_y[cc, 1] = 0
            data_y[cc, 2] = rmsd
            data_y[cc, 3] = energy
        else:
            tot = int(f.readline())
            for iiiii in range(tot):
                st = f.readline().split()
                #ix = int(st[0])
                #iy = int(st[1])
                #iz = int(st[2])
                #ic = int(st[3])
                #value = math.trunc(float(st[4]))

            continue

        tot = int(f.readline())
        for iiiii in range(tot):
            st = f.readline().split()
            ix = int(st[0])
            iy = int(st[1])
            iz = int(st[2])
            ic = int(st[3])
            value = math.trunc(float(st[4]))

            data_x[cc, ix, iy, iz, ic] = value
        cc = cc + 1
    f.close()
    fl.close()

    return data_x, data_y, cc


def read_data_from_file_easy_rmsd_rank(file_dir, data_x, data_y, true_th = 3, false_th = 3):
    f = open(file_dir+"_data", "r")

    st = f.readline().split()
    num = int(st[0])
    x = int(st[1])
    y = int(st[2])
    z = int(st[3])
    c = int(st[4])

    if (data_x.shape[0]<num or data_x.shape[4]<c):
        data_x = np.zeros((num, x, y, z, c))
        data_y = np.zeros((num, 4))

    cc = 0
    fl = open(file_dir+"_label", "r")
    data_x.fill(0)
    data_y.fill(0)
    for i in range(num):
        #if(i%10 == 0):
        #    print(i)
        st = fl.readline().split()
        rmsd = float(st[1])
        if rmsd == 0.0:
            data_y[cc, 0] = 0
            data_y[cc, 1] = 0
            data_y[cc, 2] = 1
            data_y[cc, 3] = rmsd
        elif (rmsd <= true_th):
            data_y[cc, 0] = 0
            data_y[cc, 1] = 1
            data_y[cc, 2] = 0
            data_y[cc, 3] = rmsd
        elif (rmsd > false_th):
            data_y[cc, 0] = 1
            data_y[cc, 1] = 0
            data_y[cc, 2] = 0
            data_y[cc, 3] = rmsd
        else:
            tot = int(f.readline())
            for iiiii in range(tot):
                st = f.readline().split()
                #ix = int(st[0])
                #iy = int(st[1])
                #iz = int(st[2])
                #ic = int(st[3])
                #value = math.trunc(float(st[4]))

            continue

        tot = int(f.readline())
        for iiiii in range(tot):
            st = f.readline().split()
            ix = int(st[0])
            iy = int(st[1])
            iz = int(st[2])
            ic = int(st[3])
            value = math.trunc(float(st[4]))

            data_x[cc, ix, iy, iz, ic] = value
        cc = cc + 1
    f.close()
    fl.close()

    return data_x, data_y, cc

def read_data_from_file_easy_rmsd_th(file_dir, data_x, data_y, true_th = 3, false_th = 3):
    f = open(file_dir+"_data", "r")

    st = f.readline().split()
    num = int(st[0])
    x = int(st[1])
    y = int(st[2])
    z = int(st[3])
    c = int(st[4])

    if (data_x.shape[0]<num or data_x.shape[4]<c):
        data_x = np.zeros((num, x, y, z, c))
        data_y = np.zeros((num, 3))

    cc = 0
    fl = open(file_dir+"_label", "r")
    data_x.fill(0)
    data_y.fill(0)
    for i in range(num):
        #if(i%10 == 0):
        #    print(i)
        st = fl.readline().split()
        rmsd = float(st[1])
        if (rmsd <= true_th):
            data_y[cc, 0] = 0
            data_y[cc, 1] = 1
            data_y[cc, 2] = rmsd
        elif (rmsd > false_th):
            data_y[cc, 0] = 1
            data_y[cc, 1] = 0
            data_y[cc, 2] = rmsd
        else:
            tot = int(f.readline())
            for iiiii in range(tot):
                st = f.readline().split()
                #ix = int(st[0])
                #iy = int(st[1])
                #iz = int(st[2])
                #ic = int(st[3])
                #value = math.trunc(float(st[4]))

            continue

        tot = int(f.readline())
        for iiiii in range(tot):
            st = f.readline().split()
            ix = int(st[0])
            iy = int(st[1])
            iz = int(st[2])
            ic = int(st[3])
            value = math.trunc(float(st[4]))

            data_x[cc, ix, iy, iz, ic] = value
        cc = cc + 1
    f.close()
    fl.close()

    return data_x, data_y, cc

def write_data_to_file_easy_rmsd(file_dir, data_x, data_y):
    f = open(file_dir+"_data", "w")

    size=data_x.shape
    num = size[0]
    x = size[1]
    y = size[2]
    z = size[3]
    c = size[4]
    f.write(str(num)+' '+str(x)+' '+str(y)+' '+str(z)+' '+str(c)+'\n')

    st=''
    atoms = []
    for i in range(num):
        atoms = []
        for ix in range(x):
            for iy in range(y):
                for iz in range(z):
                    for ic in range(c):
                        if (data_x[i, ix, iy, iz, ic]>0):
                            atoms.append(str(ix)+' '+str(iy)+' '+str(iz)+' '+str(ic)+' '+str(data_x[i, ix, iy, iz, ic])+'\n')
        f.write(str(len(atoms))+'\n')
        for st in atoms:
            f.write(st)


    f.close()

    f = open(file_dir+"_label", "w")
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

def read_data_from_file_easy(file_dir, data_x, data_y):
    f = open(file_dir+"_data", "r")

    st = f.readline().split()
    num = int(st[0])
    x = int(st[1])
    y = int(st[2])
    z = int(st[3])
    c = int(st[4])

    if (data_x.shape[0]<num):
        data_x = np.zeros((num, x, y, z, c))
        data_y = np.zeros((num, 2))

    data_x.fill(0)
    data_y.fill(0)
    for i in range(num):
        #if(i%10 == 0):
        #    print(i)
        tot = int(f.readline())
        for iiiii in range(tot):
            st = f.readline().split()
            ix = int(st[0])
            iy = int(st[1])
            iz = int(st[2])
            ic = int(st[3])
            value = math.trunc(float(st[4]))

            data_x[i, ix, iy, iz, ic] = value
    f.close()

    f = open(file_dir+"_label", "r")
    for i in range(num):
        st = f.readline().split()
        if(st[0] == '0'):
            data_y[i, 0] = 1
            data_y[i, 1] = 0
        else:
            data_y[i, 0] = 0
            data_y[i, 1] = 1
    f.close()

    return data_x, data_y, num

def write_data_to_file_easy(file_dir, data_x, data_y):
    f = open(file_dir+"_data", "w")

    size=data_x.shape
    num = size[0]
    x = size[1]
    y = size[2]
    z = size[3]
    c = size[4]
    f.write(str(num)+' '+str(x)+' '+str(y)+' '+str(z)+' '+str(c)+'\n')

    st=''
    atoms = []
    for i in range(num):
        atoms = []
        for ix in range(x):
            for iy in range(y):
                for iz in range(z):
                    for ic in range(c):
                        if (data_x[i, ix, iy, iz, ic]>0):
                            atoms.append(str(ix)+' '+str(iy)+' '+str(iz)+' '+str(ic)+' '+str(data_x[i, ix, iy, iz, ic])+'\n')
        f.write(str(len(atoms))+'\n')
        for st in atoms:
            f.write(st)


    f.close()

    f = open(file_dir+"_label", "w")
    for i in range(num):
        if (data_y[i, 0] == 1 and data_y[i, 1] == 0):
            f.write('0\n')
        else:
            f.write('1\n')
    f.close()

def read_data_from_file(file_dir, data_x, data_y):
    f = open(file_dir+"_data", "r")

    st = f.readline().split()
    num = int(st[0])
    x = int(st[1])
    y = int(st[2])
    z = int(st[3])
    c = int(st[4])

    if (data_x.shape[0]<num):
        data_x = np.zeros((num, x, y, z, c))
        data_y = np.zeros((num, 2))

    for i in range(num):
        #if(i%10 == 0):
        #    print(i)
        for ix in range(x):
            for iy in range(y):
                for iz in range(z):
                    st=f.readline().split()
                    for ic in range(c):
                        data_x[i,ix,iy,iz,ic] = math.trunc(float(st[ic]))

    f.close()

    f = open(file_dir+"_label", "r")
    for i in range(num):
        st = f.readline().split()
        if(st[0] == '0'):
            data_y[i, 0] = 1
            data_y[i, 1] = 0
        else:
            data_y[i, 0] = 0
            data_y[i, 1] = 1
    f.close()

    return data_x, data_y, num

def write_data_to_file(file_dir, data_x, data_y):
    f = open(file_dir+"_data", "w")

    size=data_x.shape
    num = size[0]
    x = size[1]
    y = size[2]
    z = size[3]
    c = size[4]
    f.write(str(num)+' '+str(x)+' '+str(y)+' '+str(z)+' '+str(c)+'\n')

    st=''
    for i in range(num):
        for ix in range(x):
            for iy in range(y):
                for iz in range(z):
                    st=''
                    for ic in range(c):
                        st = st + ' ' + str(data_x[i,ix,iy,iz,ic])
                    f.write(st+'\n')


    f.close()

    f = open(file_dir+"_label", "w")
    for i in range(num):
        if (data_y[i, 0] == 1 and data_y[i, 1] == 0):
            f.write('0\n')
        else:
            f.write('1\n')
    f.close()

def gen_3D_2(protein, ligand, Atoms, space, resolution, ans):

    x=0.0
    y=0.0
    z=0.0

    #ans = np.zeros((resolution, resolution, resolution, len(Atoms)))
    for line in ligand:
        #st = line.split()
        #print(line)
        #print(line[30:38],' ',line[38:46],' ',line[46:54])
        x = x + float(line[30:38])
        y = y + float(line[38:46])
        z = z + float(line[46:54])

    c_x = x / len(ligand)
    c_y = y / len(ligand)
    c_z = z / len(ligand)

    for line in protein:
        st = line.split()
        x = float(line[30:38]) - c_x
        y = float(line[38:46]) - c_y
        z = float(line[46:54]) - c_z
        atom = st[-1]

        #print(x)
        #print(y)
        #print(z)

        if ((abs(x)<=(space/2)) and (abs(y)<=(space/2)) and (abs(z)<=(space/2))):
            x = math.trunc(x/space*resolution+(resolution/2))
            y = math.trunc(y/space*resolution+(resolution/2))
            z = math.trunc(z/space*resolution+(resolution/2))
            if (x == resolution):
                x = resolution-1
            if (y == resolution):
                y = resolution-1
            if (z == resolution):
                z = resolution-1
            ans[x,y,z,Atoms.index(atom)] = ans[x,y,z,Atoms.index(atom)] + 1

    #print(ligand)
    for line in ligand:
        st = line.split()
        x = float(line[30:38]) - c_x
        y = float(line[38:46]) - c_y
        z = float(line[46:54]) - c_z
        atom = st[-1]
        
        #print(atom)
        if ((abs(x)<=(space/2)) and (abs(y)<=(space/2)) and (abs(z)<=(space/2))):
            x = math.trunc(x/space*resolution+(resolution/2))
            y = math.trunc(y/space*resolution+(resolution/2))
            z = math.trunc(z/space*resolution+(resolution/2))
            if (x == resolution):
                x = resolution-1
            if (y == resolution):
                y = resolution-1
            if (z == resolution):
                z = resolution-1
            ans[x,y,z,Atoms.index(atom)] = ans[x,y,z,Atoms.index(atom)] + 1


    return ans

def gen_3D_2_split(protein, ligand, Atoms, space, resolution, ans):

    x=0.0
    y=0.0
    z=0.0

    #ans = np.zeros((resolution, resolution, resolution, len(Atoms)))
    for line in ligand:
        #st = line.split()
        #print(line)
        #print(line[30:38],' ',line[38:46],' ',line[46:54])
        x = x + float(line[30:38])
        y = y + float(line[38:46])
        z = z + float(line[46:54])

    c_x = x / len(ligand)
    c_y = y / len(ligand)
    c_z = z / len(ligand)

    for line in protein:
        st = line.split()
        x = float(line[30:38]) - c_x
        y = float(line[38:46]) - c_y
        z = float(line[46:54]) - c_z
        atom = 'p'+st[-1]
        if atom == 'pH':
        #     print(line)
            continue

        #print(x)
        #print(y)
        #print(z)
        # print('atom:', atom)
        # print('Atoms:', Atoms)
        # print('space:', space)
        # print('resolution:', resolution)
        if ((abs(x)<=(space/2)) and (abs(y)<=(space/2)) and (abs(z)<=(space/2))):
            x = math.trunc(x/space*resolution+(resolution/2))
            y = math.trunc(y/space*resolution+(resolution/2))
            z = math.trunc(z/space*resolution+(resolution/2))
            if (x == resolution):
                x = resolution-1
            if (y == resolution):
                y = resolution-1
            if (z == resolution):
                z = resolution-1
            # print("x:", x)
            # print("y:", x)
            # print("z:", x)
            ans[x,y,z,Atoms.index(atom)] = ans[x,y,z,Atoms.index(atom)] + 1

    #print(ligand)
    for line in ligand:
        st = line.split()
        x = float(line[30:38]) - c_x
        y = float(line[38:46]) - c_y
        z = float(line[46:54]) - c_z
        atom = 'l'+st[-1]
        # if atom == 'lH':
        #     print(line)
        #     continue
        
        #print(atom)
        if ((abs(x)<=(space/2)) and (abs(y)<=(space/2)) and (abs(z)<=(space/2))):
            x = math.trunc(x/space*resolution+(resolution/2))
            y = math.trunc(y/space*resolution+(resolution/2))
            z = math.trunc(z/space*resolution+(resolution/2))
            if (x == resolution):
                x = resolution-1
            if (y == resolution):
                y = resolution-1
            if (z == resolution):
                z = resolution-1
            ans[x,y,z,Atoms.index(atom)] = ans[x,y,z,Atoms.index(atom)] + 1


    return ans

def read_DUDE_to_disk(input_list, DUDE_dir, pdb_dir, output_dir, resolution, tile_size):

    rec_list = []
    f_list = open(input_list, "r")
    for line in f_list:
        rec_list.append(line[:-1])
    f_list.close()

    Atoms = []
    actives_n = []
    decoys_n = []
    actives_lists = []
    decoys_lists = []

    """count proteins Atoms type

    """
    receptor_num = 0
    for line in rec_list:
        #ligand_list = os.listdir(DUDE_dir+"/"+line+"/ac")
        f = open(DUDE_dir+"/all/"+line+"/receptor_rcsb.pdb","r")
        #print("/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/medusa/output/"+line+".pdb")
        for st in f:
            ss = st.split()
            #print(ss)
            #if (ss[0] == 'REMARK') and (ss[1] == 'POSE:'):
            #    receptor_num = receptor_num + 1
            if (ss[0] == 'ATOM') or (ss[0] == 'HETATM'):
                #atom = ss[2][0]
                #if(atom >='0') and (atom <= '9'):
                #    atom = ss[2][1]
                atom = ss[-1]
                if not atom in Atoms:
                    Atoms.append(atom)

        f.close()


    """count ligands Atoms type

    """
    #neg_receptor_num = 1500
    #f = open(input_dir+"/receptor_2.pdb","r")
    #for st in f:
    #    ss = st.split()
    #    if (ss[0] == 'ATOM') or (ss[0] == 'HETATM'):
    #        atom = ss[-1]
    #        if not atom in Atoms:
    #            Atoms.append(atom)
    #f.close()

    pos_data = 0
    neg_data = 0
    for line in rec_list:
        f = open(DUDE_dir+"/all/"+line+"/actives_final.mol2","r")
        tt = 0
        ligand_list=os.listdir(pdb_dir+"/"+line+"/actives/")
        cc = len(ligand_list)
        for st in f:
            ss = st.split()
            if (len(ss) <= 0):
                continue
            if (ss[0] == '@<TRIPOS>ATOM'):
                tt = 1
                continue
            if (ss[0] == '@<TRIPOS>BOND'):
                tt = 0
                continue
            if (tt == 1):
                atom = ss[1]
                while(atom[0] >= '0' and atom[0] <= '9'):
                    atom = atom[1:]
                while(atom[-1] >= '0' and atom[-1] <= '9'):
                    atom = atom[:-1]
                if not atom in Atoms:
                    Atoms.append(atom)
        f.close()
        pos_data = pos_data + cc
        actives_n.append(cc)
        actives_lists.append(ligand_list)


        f = open(DUDE_dir+"/all/"+line+"/decoys_final.mol2","r")
        tt = 0
        ligand_list=os.listdir(pdb_dir+"/"+line+"/decoys/")
        cc = len(ligand_list)
        for st in f:
            ss = st.split()
            if (len(ss) <= 0):
                continue
            if (ss[0] == '@<TRIPOS>ATOM'):
                tt = 1
                continue
            if (ss[0] == '@<TRIPOS>BOND'):
                tt = 0
                continue
            if (tt == 1):
                atom = ss[1]
                while(atom[0] >= '0' and atom[0] <= '9'):
                    atom = atom[1:]
                while(atom[-1] >= '0' and atom[-1] <= '9'):
                    atom = atom[:-1]
                if not atom in Atoms:
                    Atoms.append(atom)
        f.close()
        neg_data = neg_data + cc
        decoys_n.append(cc)
        decoys_lists.append(ligand_list)

        
    #neg_data = 3000
    print("total data: " + str(pos_data+neg_data))
    print("total atoms: " + str(len(Atoms)))


    """Get data list

    """
    total_data_list = []
    for p in rec_list:
        f = open(pdb_dir+"/"+p+"/actives_list","r")
        for line in f:
            total_data_list.append(p+"/actives/"+line.split()[0])
        f.close()

        f = open(pdb_dir+"/"+p+"/decoys_list","r")
        for line in f:
            total_data_list.append(p+"/decoys/"+line.split()[0])
        f.close()


    """Generate 3D grids

    """
    tot = len(total_data_list)
    tt = list(np.arange(tot))
    random_list = random.sample(tt, tot)

    tmp_data = np.zeros((tile_size, resolution, resolution, resolution, len(Atoms)))
    tmp_label = np.zeros((tile_size, 2))

    tot = 0
    file_counter = 0
    for i in range(len(total_data_list)):
        if ('actives' in total_data_list[random_list[i]]):
            tmp_label[tot, 0] = 0
            tmp_label[tot, 1] = 1
        else:
            tmp_label[tot, 0] = 1
            tmp_label[tot, 1] = 0

        f = open(pdb_dir+"/"+total_data_list[random_list[i]],"r")
        for st in f:
            ss = st.split()
            if (ss[0] == 'ENDMDL'):
                #print(iiiii)
                #tmp_data[tot] = gen_3D_2(protein, ligand, Atoms, 20, resolution)
                tmp_data[tot] = gen_3D_2(protein, ligand, Atoms, 20, resolution, tmp_data[tot])
                tot = tot + 1
                protein = []
                ligand = []
                break
            if (ss[0] == 'ATOM'):
                protein.append(st[:-1])
            if (ss[0] == 'HETATM'):
                ligand.append(st[:-1])

        f.close()

        if (tot >= tile_size):
            write_data_to_file(output_dir+"/"+str(file_counter), tmp_data, tmp_label)
            file_counter = file_counter + 1
            tot = 0
            tmp_data.fill(0)
            tmp_label.fill(0)

    if (tot>0):
        write_data_to_file(output_dir+"/"+str(file_counter), tmp_data, tmp_label)
        file_counter = file_counter + 1
        tot = 0
        tmp_data.fill(0)
        tmp_label.fill(0)

def read_pdbbind_to_disk(input_list, label_list_file, pdbbind_dir, groundtruth_dir, output_dir, resolution, tile_size):

    rec_list = []
    label_list = []
    f_list = open(input_list, "r")
    for line in f_list:
        rec_list.append(line.split()[0])
        #label_list.append(line.split()[1])
    f_list.close()

    label_file_list = []
    f_list = open(label_list_file, "r")
    for line in f_list:
        rmsd = float(line.split()[0])
        if (rmsd < 3):
            label_file_list.append(1)
        else:
            label_file_list.append(0)
    f_list.close()

    #Atoms = []
    Atoms = ['N', 'C', 'O', 'H', 'S', 'Br', 'Cl', 'P', 'F', 'I']
    actives_n = []
    decoys_n = []
    actives_lists = []
    decoys_lists = []

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
                if not atom in Atoms:
                    Atoms.append(atom)

        f.close()

        
    #neg_data = 3000
    print("total data: " + str(tot))
    print("total poses: " + str(poses_num))
    print("total atoms: " + str(len(Atoms)))
    print(Atoms)


    """Generate 3D data files

    """
    #tot = pos_data + neg_data
    kongge = '          '
    kongge = kongge + kongge
    kongge = kongge + kongge
    kongge = kongge + kongge


    tmp_data = np.zeros((tile_size, resolution, resolution, resolution, len(Atoms)))
    tmp_label = np.zeros((tile_size, 2))

    tot = len(rec_list)
    tt = list(np.arange(tot))
    random_list = random.sample(tt, tot)

    tot = 0
    protein = []
    ligand = []
    cc = 0
    global_tot = 0
    file_counter = 0
    for i in range(len(rec_list)):
        #line = rec_list[random_list[i]]
        #if (label_list[random_list[i]] == 1):
        #    tmp_label[tot, 0] = 0
        #    tmp_label[tot, 1] = 1
        #else:
        #    tmp_label[tot, 0] = 1
        #    tmp_label[tot, 1] = 0
        line = rec_list[i]

        f = open(pdbbind_dir+"/"+line,"r")
        flag_label = 0
        for st in f:
            ss = st.split()
            if (ss[0] == 'REMARK') and (ss[1] == 'POSE:'):
                if(flag_label == 0):
                    print(i,' ',global_tot)
                    flag_label = 1
                global_tot = global_tot + 1
            if (ss[0] == 'ENDMDL'):
                tmp_data[tot] = gen_3D_2(protein, ligand, Atoms, 20, resolution, tmp_data[tot])
                protein = []
                ligand = []
                if (label_file_list[cc] == 1):
                    tmp_label[tot, 0] = 0
                    tmp_label[tot, 1] = 1
                else:
                    tmp_label[tot, 0] = 1
                    tmp_label[tot, 1] = 0
                tot = tot + 1
                cc = cc + 1

                if (tot >= tile_size):
                    write_data_to_file_easy(output_dir+"/"+str(file_counter), tmp_data, tmp_label)
                    file_counter = file_counter + 1
                    tot = 0
                    tmp_data.fill(0)
                    tmp_label.fill(0)
                
            if (ss[0] == 'ATOM'):
                protein.append(st[:-1])
            if (ss[0] == 'HETATM'):
                ligand.append(st[:-1])

        f.close()

        """add ground truth

        """
        protein = []
        ligand = []
        f = open(groundtruth_dir+'/'+line[:4]+'/'+line[:4]+'.rec.pdb')
        for st in f:
            ss = st.split()
            if (ss[0] == 'ATOM') or (ss[0] == 'HETATM'):
                protein.append(st[:-1])
        f.close()

        f = open(groundtruth_dir+'/'+line[:4]+'/'+line[:4]+'.lig.mol2')
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
                ligand.append(ss)
                #print(st)
        f.close()
        tmp_data[tot] = gen_3D_2(protein, ligand, Atoms, 20, resolution, tmp_data[tot])
        tmp_label[tot, 0] = 0
        tmp_label[tot, 1] = 1
        tot = tot + 1
        protein = []
        ligand = []
        if (tot >= tile_size):
            write_data_to_file_easy(output_dir+"/"+str(file_counter), tmp_data, tmp_label)
            file_counter = file_counter + 1
            tot = 0
            tmp_data.fill(0)
            tmp_label.fill(0)


    if (tot>0):
        print('tot:', tot)
        write_data_to_file_easy(output_dir+"/"+str(file_counter), tmp_data[:tot,:,:,:,:], tmp_label[:tot,:])
        file_counter = file_counter + 1
        tot = 0
        tmp_data.fill(0)
        tmp_label.fill(0)

    print("3D data generated")



def read_pdbbind_to_disk_rmsd(input_list, label_list_file, pdbbind_dir, groundtruth_dir, output_dir, resolution, tile_size):

    rec_list = []
    label_list = []
    f_list = open(input_list, "r")
    for line in f_list:
        rec_list.append(line.split()[0])
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
    Atoms = ['N', 'C', 'O', 'H', 'S', 'Br', 'Cl', 'P', 'F', 'I']
    actives_n = []
    decoys_n = []
    actives_lists = []
    decoys_lists = []

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
                if not atom in Atoms:
                    Atoms.append(atom)

        f.close()

        
    #neg_data = 3000
    print("total data: " + str(tot))
    print("total poses: " + str(poses_num))
    print("total atoms: " + str(len(Atoms)))
    print(Atoms)


    """Generate 3D data files

    """
    #tot = pos_data + neg_data
    kongge = '          '
    kongge = kongge + kongge
    kongge = kongge + kongge
    kongge = kongge + kongge


    tmp_data = np.zeros((tile_size, resolution, resolution, resolution, len(Atoms)))
    tmp_label = np.zeros((tile_size, 3))

    tot = len(rec_list)
    tt = list(np.arange(tot))
    random_list = random.sample(tt, tot)

    tot = 0
    protein = []
    ligand = []
    cc = 0
    global_tot = 0
    file_counter = 0
    for i in range(len(rec_list)):
        line = rec_list[i]

        f = open(pdbbind_dir+"/"+line,"r")
        flag_label = 0
        for st in f:
            ss = st.split()
            if (ss[0] == 'REMARK') and (ss[1] == 'POSE:'):
                if(flag_label == 0):
                    print(i,' ',global_tot)
                    flag_label = 1
                global_tot = global_tot + 1
            if (ss[0] == 'ENDMDL'):
                tmp_data[tot] = gen_3D_2(protein, ligand, Atoms, 20, resolution, tmp_data[tot])
                protein = []
                ligand = []
                if (label_file_list[cc] == 1):
                    tmp_label[tot, 0] = 0
                    tmp_label[tot, 1] = 1
                else:
                    tmp_label[tot, 0] = 1
                    tmp_label[tot, 1] = 0
                tmp_label[tot, 2] = label_file_list_rmsd[cc]
                tot = tot + 1
                cc = cc + 1

                if (tot >= tile_size):
                    write_data_to_file_easy_rmsd(output_dir+"/"+str(file_counter), tmp_data, tmp_label)
                    file_counter = file_counter + 1
                    tot = 0
                    tmp_data.fill(0)
                    tmp_label.fill(0)
                
            if (ss[0] == 'ATOM'):
                protein.append(st[:-1])
            if (ss[0] == 'HETATM'):
                ligand.append(st[:-1])

        f.close()

        """add ground truth

        """
        protein = []
        ligand = []
        f = open(groundtruth_dir+'/'+line[:4]+'/'+line[:4]+'.rec.pdb')
        for st in f:
            ss = st.split()
            if (ss[0] == 'ATOM') or (ss[0] == 'HETATM'):
                protein.append(st[:-1])
        f.close()

        f = open(groundtruth_dir+'/'+line[:4]+'/'+line[:4]+'.lig.mol2')
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
                ligand.append(ss)
                #print(st)
        f.close()
        tmp_data[tot] = gen_3D_2(protein, ligand, Atoms, 20, resolution, tmp_data[tot])
        tmp_label[tot, 0] = 0
        tmp_label[tot, 1] = 1
        tmp_label[tot, 2] = 0.0
        tot = tot + 1
        protein = []
        ligand = []
        if (tot >= tile_size):
            write_data_to_file_easy_rmsd(output_dir+"/"+str(file_counter), tmp_data, tmp_label)
            file_counter = file_counter + 1
            tot = 0
            tmp_data.fill(0)
            tmp_label.fill(0)


    if (tot>0):
        print('tot:', tot)
        write_data_to_file_easy_rmsd(output_dir+"/"+str(file_counter), tmp_data[:tot,:,:,:,:], tmp_label[:tot,:])
        file_counter = file_counter + 1
        tot = 0
        tmp_data.fill(0)
        tmp_label.fill(0)

    print("3D data generated")


def read_pdbbind_to_disk_rmsd_split(input_list, label_list_file, pdbbind_dir, groundtruth_dir, output_dir, resolution, tile_size):

    rec_list = []
    label_list = []
    f_list = open(input_list, "r")
    for line in f_list:
        rec_list.append(line.split()[0])
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


    tmp_data = np.zeros((tile_size, resolution, resolution, resolution, len(Atoms)))
    tmp_label = np.zeros((tile_size, 3))

    tot = len(rec_list)
    tt = list(np.arange(tot))
    random_list = random.sample(tt, tot)

    tot = 0
    protein = []
    ligand = []
    cc = 0
    global_tot = 0
    file_counter = 0
    for i in range(len(rec_list)):
        H_type = []

        line = rec_list[i]

        f = open(pdbbind_dir+"/"+line,"r")
        flag_label = 0
        for st in f:
            ss = st.split()
            if (ss[0] == 'REMARK') and (ss[1] == 'POSE:'):
                if(flag_label == 0):
                    print(i,' ',global_tot)
                    flag_label = 1
                global_tot = global_tot + 1
            if (ss[0] == 'ENDMDL'):
                tmp_data[tot] = gen_3D_2_split(protein, ligand, Atoms, 20, resolution, tmp_data[tot])
                protein = []
                ligand = []
                if (label_file_list[cc] == 1):
                    tmp_label[tot, 0] = 0
                    tmp_label[tot, 1] = 1
                else:
                    tmp_label[tot, 0] = 1
                    tmp_label[tot, 1] = 0
                tmp_label[tot, 2] = label_file_list_rmsd[cc]
                tot = tot + 1
                cc = cc + 1

                if (tot >= tile_size):
                    write_data_to_file_easy_rmsd(output_dir+"/"+str(file_counter), tmp_data, tmp_label)
                    file_counter = file_counter + 1
                    tot = 0
                    tmp_data.fill(0)
                    tmp_label.fill(0)
                
            if (ss[0] == 'ATOM'):
                protein.append(st[:-1])
            if (ss[0] == 'HETATM'):
                ligand.append(st[:-1])
                if (ss[-1] == 'H') and (not ss[-1] in H_type):
                    H_type.append(ss[2])

        f.close()

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
        tmp_data[tot] = gen_3D_2_split(protein, ligand, Atoms, 20, resolution, tmp_data[tot])
        tmp_label[tot, 0] = 0
        tmp_label[tot, 1] = 1
        tmp_label[tot, 2] = 0.0
        tot = tot + 1
        protein = []
        ligand = []
        if (tot >= tile_size):
            write_data_to_file_easy_rmsd(output_dir+"/"+str(file_counter), tmp_data, tmp_label)
            file_counter = file_counter + 1
            tot = 0
            tmp_data.fill(0)
            tmp_label.fill(0)


    if (tot>0):
        print('tot:', tot)
        write_data_to_file_easy_rmsd(output_dir+"/"+str(file_counter), tmp_data[:tot,:,:,:,:], tmp_label[:tot,:])
        file_counter = file_counter + 1
        tot = 0
        tmp_data.fill(0)
        tmp_label.fill(0)

    print("3D data generated")

def read_pdbbind_to_disk_rmsd_energy_split(input_list, label_list_file, pdbbind_dir, groundtruth_dir, output_dir, resolution, tile_size):

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


    tmp_data = np.zeros((tile_size, resolution, resolution, resolution, len(Atoms)))
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
    for i in range(len(rec_list)):
        H_type = []

        line = rec_list[i]

        f = open(pdbbind_dir+"/"+line,"r")
        flag_label = 0
        for st in f:
            ss = st.split()
            if (ss[0] == 'REMARK') and (ss[1] == 'POSE:'):
                if(flag_label == 0):
                    print(i,' ',global_tot)
                    flag_label = 1
                global_tot = global_tot + 1
            if (ss[0] == 'REMARK') and (ss[1] == 'E_total:'):
                this_pose_energy = float(ss[2])
            if (ss[0] == 'ENDMDL'):
                tmp_data[tot] = gen_3D_2_split(protein, ligand, Atoms, 20, resolution, tmp_data[tot])
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
                    write_data_to_file_easy_rmsd(output_dir+"/"+str(file_counter), tmp_data, tmp_label)
                    file_counter = file_counter + 1
                    tot = 0
                    tmp_data.fill(0)
                    tmp_label.fill(0)
                
            if (ss[0] == 'ATOM'):
                protein.append(st[:-1])
            if (ss[0] == 'HETATM'):
                ligand.append(st[:-1])
                if (ss[-1] == 'H') and (not ss[-1] in H_type):
                    H_type.append(ss[2])

        f.close()

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
        tmp_data[tot] = gen_3D_2_split(protein, ligand, Atoms, 20, resolution, tmp_data[tot])
        tmp_label[tot, 0] = 0
        tmp_label[tot, 1] = 1
        tmp_label[tot, 2] = 0.0
        tmp_label[tot, 3] = -41.7
        tot = tot + 1
        protein = []
        ligand = []
        if (tot >= tile_size):
            write_data_to_file_easy_rmsd(output_dir+"/"+str(file_counter), tmp_data, tmp_label)
            file_counter = file_counter + 1
            tot = 0
            tmp_data.fill(0)
            tmp_label.fill(0)
        


    if (tot>0):
        print('tot:', tot)
        write_data_to_file_easy_rmsd(output_dir+"/"+str(file_counter), tmp_data[:tot,:,:,:,:], tmp_label[:tot,:])
        file_counter = file_counter + 1
        tot = 0
        tmp_data.fill(0)
        tmp_label.fill(0)

    print("3D data generated")
