# python convert_data_to_disk.py /gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/AtomNet/tmp_data/pdbbind_ /gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/AtomNet/tmp_data/pdbbind_ /gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/medusa/pdbbind_output/ /gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/pdbbind/ /gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/AtomNet/tmp_data/pdbbind_rmsd_resolution30 30
# python convert_data_to_disk.py --cv=0 --output_file=pdbbind_rmsd_srand200 --start_iter=3 --end_iter=7 --thread_num=4 --use_new_data

#import tensorflow as tf
import numpy as np
import os
import sys
#from sklearn import metrics

import multiprocessing as mp

import dataset_from_file as dff
import gcn_data as gd
import coordinate_data as cd
import coordinate_data2 as cd2
import screen_data as sd
import muv_data as md
import coordinate_data_15844 as cd15844

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cv", help="cross-validation fold", type=int, default = 0)
parser.add_argument("--resolution", help="resolution of the pose", type=int, default = 20)
parser.add_argument("--bond_th", help="create a bond for the pair of atoms which distance less than bond_th", type=int, default = 6)
parser.add_argument("--pocket_th", help="the threshold of distance to the centroid of ligand to be considered as pocket", type=float, default = 12)
parser.add_argument("--output_file", help="output file name of this train/test data", type=str, default = None)
parser.add_argument("--start_iter", help="create training data from which random seed", type=int, default = 3)
parser.add_argument("--end_iter", help="create training data till which random seed", type=int, default = 14)
parser.add_argument("--thread_num", help="num of threads to creating dataset", type=int, default = 4)
parser.add_argument("--use_new_data", help="create data for predicting 3D coordinate", default=False, action='store_true')
parser.add_argument("--screen_data", help="If we generate data for screen", default=False, action='store_true')
parser.add_argument("--pdbbind_dir", help="dir to the pdbbind dataset output", type=str, default='/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/medusa/pdbbind_output_s')
parser.add_argument("--label_list_file", help="the path to label files", type=str, default='/gpfs/group/mtk2/cyberstar/hzj5142/GNN/GNN/DGNN/data/pdbbind')
parser.add_argument("--input_list", help="list of train/test pdbs", type=str, default='/gpfs/group/mtk2/cyberstar/hzj5142/GNN/GNN/DGNN/data/pdbbind/pdb_list_')
parser.add_argument("--groundtruth_dir", help="the path to the ground truth pose pdbbind files", type=str, default='/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/pdbbind/')
parser.add_argument("--dataset", help="type of dataset", type=str, default='screen_data')
parser.add_argument("--pdb_version", help="version of pdbbind", type=int, default = 2016)
parser.add_argument("--muv_dir", help="path to muv dataset", type=str, default='/gpfs/group/mtk2/cyberstar/hzj5142/GNN/GNN/DGNN/data/MUV/origin')
parser.add_argument("--muv_target", help="target id of muv data", type=str, default='466')
parser.add_argument("--muv_label", help="which label of the muv data", type=str, default='decoy')
parser.add_argument("--casf_groundtruth_dir", help="path to casf dataset GT", type=str, default='/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/CASF-2016/coreset')
parser.add_argument("--casf_dir", help="path to casf dataset", type=str, default='/gpfs/scratch/hzj5142/AtomNet/CASF/s367')
args = parser.parse_args()
print(args)


def load_and_save_dataset(input_list, label_list_file, pdbbind_dir, groundtruth_dir, output_dir, resolution):
    tile_size = 1024

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if not os.path.isdir(output_dir+'/train'):
        os.makedirs(output_dir+'/train')
    dff.read_pdbbind_to_disk_rmsd_split(input_list+'train', label_list_file+'train_label', pdbbind_dir, groundtruth_dir,
      output_dir+'/train', resolution, tile_size)

    if not os.path.isdir(output_dir+'/test'):
        os.makedirs(output_dir+'/test')
    dff.read_pdbbind_to_disk_rmsd_split(input_list+'test', label_list_file+'test_label', pdbbind_dir, groundtruth_dir,
      output_dir+'/test', resolution, tile_size)


def srand_data_load_save(input_list, label_list_file, pdbbind_dir, groundtruth_dir, output_dir, cv, resolution, iteration):
    tile_size = 1024

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(output_dir+'/train'):
        os.makedirs(output_dir+'/train')
    if not os.path.isdir(output_dir+'/test'):
        os.makedirs(output_dir+'/test')

    output_dir_tmp = output_dir + '_tmp'
    if not os.path.isdir(output_dir_tmp):
        os.makedirs(output_dir_tmp)
    if not os.path.isdir(output_dir_tmp+'/train'):
        os.makedirs(output_dir_tmp+'/train')
    if not os.path.isdir(output_dir_tmp+'/test'):
        os.makedirs(output_dir_tmp+'/test')

    train_file_num = 0
    test_file_num = 0
    for i in range(3,14):
        '''
        dff.read_pdbbind_to_disk_rmsd_energy_split(input_list+'train', label_list_file+'/train_sb_s'+str(i)+'67.test', pdbbind_dir+str(i)+'67', groundtruth_dir,
          output_dir_tmp+'/train', resolution, tile_size)
        dataset_file_list = os.listdir(output_dir_tmp+'/train')
        print(dataset_file_list)

        n = len(dataset_file_list) // 2
        for j in range(train_file_num, train_file_num + n):
            dataset = output_dir_tmp+'/train/'+str(j-train_file_num)+'_data'
            os.rename(dataset, output_dir+'/train/'+str(j)+'_data')
            label = output_dir_tmp+'/train/'+str(j-train_file_num)+'_label'
            os.rename(label, output_dir+'/train/'+str(j)+'_label')
        train_file_num = train_file_num + n
        '''



        dff.read_pdbbind_to_disk_rmsd_energy_split(input_list+'test', label_list_file+'/test_sb_s'+str(i)+'67.test', pdbbind_dir+str(i)+'67', groundtruth_dir,
          output_dir_tmp+'/test', resolution, tile_size)
        dataset_file_list = os.listdir(output_dir_tmp+'/test')
        print(dataset_file_list)

        n = len(dataset_file_list) // 2
        for j in range(test_file_num, test_file_num + n):
            dataset = output_dir_tmp+'/test/'+str(j-test_file_num)+'_data'
            os.rename(dataset, output_dir+'/test/'+str(j)+'_data')
            label = output_dir_tmp+'/test/'+str(j-test_file_num)+'_label'
            os.rename(label, output_dir+'/test/'+str(j)+'_label')
        test_file_num = test_file_num + n


def srand_data_load_save_gcn_thread(input_list, label_list_file, pdbbind_dir, groundtruth_dir, output_dir, cv, resolution, iteration, start, end, train_fn, test_fn, thread_num, thread_id):
    tile_size = 1024
    output_dir_tmp = output_dir + '_tmp_' + str(thread_id)
    if not os.path.isdir(output_dir_tmp):
        os.makedirs(output_dir_tmp)
    if not os.path.isdir(output_dir_tmp+'/train'):
        os.makedirs(output_dir_tmp+'/train')
    if not os.path.isdir(output_dir_tmp+'/test'):
        os.makedirs(output_dir_tmp+'/test')

    for i in range(start + thread_id, end, thread_num):
        print("seed " + str(i))

        
        gd.read_pdbbind_to_disk_rmsd_energy_split(input_list+'train', label_list_file+'/train_sb_s'+str(i)+'67.test', pdbbind_dir+str(i)+'67', groundtruth_dir,
          output_dir_tmp+'/train', resolution, tile_size, seed = i)
        dataset_file_list = os.listdir(output_dir_tmp+'/train')
        print(dataset_file_list)

        n = len(dataset_file_list) // 3
        train_file_num = train_fn + n * (i - start)
        for j in range(train_file_num, train_file_num + n):
            dataset = output_dir_tmp+'/train/'+str(j-train_file_num)+'_data-feats'
            os.rename(dataset, output_dir+'/train/'+str(j)+'_data-feats')
            dataset = output_dir_tmp+'/train/'+str(j-train_file_num)+'_data-G.json'
            os.rename(dataset, output_dir+'/train/'+str(j)+'_data-G.json')
            label = output_dir_tmp+'/train/'+str(j-train_file_num)+'_label'
            os.rename(label, output_dir+'/train/'+str(j)+'_label')
        #train_file_num = train_file_num + n
        
        
        gd.read_pdbbind_to_disk_rmsd_energy_split(input_list+'test', label_list_file+'/test_sb_s'+str(i)+'67.test', pdbbind_dir+str(i)+'67', groundtruth_dir,
          output_dir_tmp+'/test', resolution, tile_size, seed = i)
        dataset_file_list = os.listdir(output_dir_tmp+'/test')
        print(dataset_file_list)

        n = len(dataset_file_list) // 3
        test_file_num = test_fn + n * (i - start)
        for j in range(test_file_num, test_file_num + n):
            dataset = output_dir_tmp+'/test/'+str(j-test_file_num)+'_data-feats'
            os.rename(dataset, output_dir+'/test/'+str(j)+'_data-feats')
            dataset = output_dir_tmp+'/test/'+str(j-test_file_num)+'_data-G.json'
            os.rename(dataset, output_dir+'/test/'+str(j)+'_data-G.json')
            label = output_dir_tmp+'/test/'+str(j-test_file_num)+'_label'
            os.rename(label, output_dir+'/test/'+str(j)+'_label')
        #test_file_num = test_file_num + n


def srand_data_load_save_gcn(input_list, label_list_file, pdbbind_dir, groundtruth_dir, output_dir, cv, resolution, iteration, start, end, thread_num = 1):
    tile_size = 1024
    print("srand_data_load_save_gcn")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(output_dir+'/train'):
        os.makedirs(output_dir+'/train')
    if not os.path.isdir(output_dir+'/test'):
        os.makedirs(output_dir+'/test')

    output_dir_tmp = output_dir + '_tmp'
    if not os.path.isdir(output_dir_tmp):
        os.makedirs(output_dir_tmp)
    if not os.path.isdir(output_dir_tmp+'/train'):
        os.makedirs(output_dir_tmp+'/train')
    if not os.path.isdir(output_dir_tmp+'/test'):
        os.makedirs(output_dir_tmp+'/test')

    print("data dir created!")

    train_file_num = len(os.listdir(output_dir+'/train')) // 3
    test_file_num = len(os.listdir(output_dir+'/test')) // 3
    # for i in range(4,5):
    if thread_num == 1:
        for i in range(start, end):
            print("seed " + str(i))

            
            gd.read_pdbbind_to_disk_rmsd_energy_split(input_list+'train', label_list_file+'/train_sb_s'+str(i)+'67.test', pdbbind_dir+str(i)+'67', groundtruth_dir,
              output_dir_tmp+'/train', resolution, tile_size, seed = i)
            dataset_file_list = os.listdir(output_dir_tmp+'/train')
            print(dataset_file_list)

            n = len(dataset_file_list) // 3
            for j in range(train_file_num, train_file_num + n):
                dataset = output_dir_tmp+'/train/'+str(j-train_file_num)+'_data-feats'
                os.rename(dataset, output_dir+'/train/'+str(j)+'_data-feats')
                dataset = output_dir_tmp+'/train/'+str(j-train_file_num)+'_data-G.json'
                os.rename(dataset, output_dir+'/train/'+str(j)+'_data-G.json')
                label = output_dir_tmp+'/train/'+str(j-train_file_num)+'_label'
                os.rename(label, output_dir+'/train/'+str(j)+'_label')
            train_file_num = train_file_num + n
            
            
            gd.read_pdbbind_to_disk_rmsd_energy_split(input_list+'test', label_list_file+'/test_sb_s'+str(i)+'67.test', pdbbind_dir+str(i)+'67', groundtruth_dir,
              output_dir_tmp+'/test', resolution, tile_size, seed = i)
            dataset_file_list = os.listdir(output_dir_tmp+'/test')
            print(dataset_file_list)

            n = len(dataset_file_list) // 3
            for j in range(test_file_num, test_file_num + n):
                dataset = output_dir_tmp+'/test/'+str(j-test_file_num)+'_data-feats'
                os.rename(dataset, output_dir+'/test/'+str(j)+'_data-feats')
                dataset = output_dir_tmp+'/test/'+str(j-test_file_num)+'_data-G.json'
                os.rename(dataset, output_dir+'/test/'+str(j)+'_data-G.json')
                label = output_dir_tmp+'/test/'+str(j-test_file_num)+'_label'
                os.rename(label, output_dir+'/test/'+str(j)+'_label')
            test_file_num = test_file_num + n
    else:
        p_list = []
        for thread_id in range(thread_num):
            p = mp.Process(target=srand_data_load_save_gcn_thread, args=(input_list, label_list_file, pdbbind_dir, groundtruth_dir, output_dir, cv, resolution, iteration, start, end, train_file_num, test_file_num, thread_num, thread_id))
            p.start()
            p_list.append(p)
    
        for p in p_list:
            p.join()


def srand_data_load_save_coord_thread(input_list, label_list_file, pdbbind_dir, groundtruth_dir, output_dir, cv, resolution, bond_th, iteration, start, end, train_fn, test_fn, thread_num, thread_id):
    tile_size = 1024
    output_dir_tmp = output_dir + '_tmp_' + str(thread_id)
    if not os.path.isdir(output_dir_tmp):
        os.makedirs(output_dir_tmp)
    if not os.path.isdir(output_dir_tmp+'/train'):
        os.makedirs(output_dir_tmp+'/train')
    if not os.path.isdir(output_dir_tmp+'/test'):
        os.makedirs(output_dir_tmp+'/test')

    for i in range(start + thread_id, end, thread_num):
        print("seed " + str(i))

        
        cd.read_pdbbind_to_disk_rmsd_energy_split(input_list+'train', label_list_file+'/train_sb_s'+str(i)+'67.test', pdbbind_dir+str(i)+'67', groundtruth_dir,
          output_dir_tmp+'/train', resolution, tile_size, bond_th, seed = i)
        dataset_file_list = os.listdir(output_dir_tmp+'/train')
        print(dataset_file_list)

        n = len(dataset_file_list) // 3
        train_file_num = train_fn + n * (i - start)
        for j in range(train_file_num, train_file_num + n):
            dataset = output_dir_tmp+'/train/'+str(j-train_file_num)+'_data-feats'
            os.rename(dataset, output_dir+'/train/'+str(j)+'_data-feats')
            dataset = output_dir_tmp+'/train/'+str(j-train_file_num)+'_data-G.json'
            os.rename(dataset, output_dir+'/train/'+str(j)+'_data-G.json')
            label = output_dir_tmp+'/train/'+str(j-train_file_num)+'_label'
            os.rename(label, output_dir+'/train/'+str(j)+'_label')
        #train_file_num = train_file_num + n
        
        
        
        cd.read_pdbbind_to_disk_rmsd_energy_split(input_list+'test', label_list_file+'/test_sb_s'+str(i)+'67.test', pdbbind_dir+str(i)+'67', groundtruth_dir,
          output_dir_tmp+'/test', resolution, tile_size, bond_th, seed = i)
        dataset_file_list = os.listdir(output_dir_tmp+'/test')
        print(dataset_file_list)

        n = len(dataset_file_list) // 3
        test_file_num = test_fn + n * (i - start)
        for j in range(test_file_num, test_file_num + n):
            dataset = output_dir_tmp+'/test/'+str(j-test_file_num)+'_data-feats'
            os.rename(dataset, output_dir+'/test/'+str(j)+'_data-feats')
            dataset = output_dir_tmp+'/test/'+str(j-test_file_num)+'_data-G.json'
            os.rename(dataset, output_dir+'/test/'+str(j)+'_data-G.json')
            label = output_dir_tmp+'/test/'+str(j-test_file_num)+'_label'
            os.rename(label, output_dir+'/test/'+str(j)+'_label')
        

def srand_data_load_save_coord(input_list, label_list_file, pdbbind_dir, groundtruth_dir, output_dir, cv, resolution, bond_th, iteration, start, end, thread_num = 1):
    tile_size = 1024
    print("srand_data_load_save_coord")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(output_dir+'/train'):
        os.makedirs(output_dir+'/train')
    if not os.path.isdir(output_dir+'/test'):
        os.makedirs(output_dir+'/test')

    output_dir_tmp = output_dir + '_tmp'
    if not os.path.isdir(output_dir_tmp):
        os.makedirs(output_dir_tmp)
    if not os.path.isdir(output_dir_tmp+'/train'):
        os.makedirs(output_dir_tmp+'/train')
    if not os.path.isdir(output_dir_tmp+'/test'):
        os.makedirs(output_dir_tmp+'/test')

    print("data dir created!")

    train_file_num = len(os.listdir(output_dir+'/train')) // 3
    test_file_num = len(os.listdir(output_dir+'/test')) // 3
    # for i in range(4,5):
    if thread_num == 1:
        for i in range(start, end):
            print("seed " + str(i))

            
            cd.read_pdbbind_to_disk_rmsd_energy_split(input_list+'train', label_list_file+'/train_sb_s'+str(i)+'67.test', pdbbind_dir+str(i)+'67', groundtruth_dir,
              output_dir_tmp+'/train', resolution, tile_size, bond_th, seed = i)
            dataset_file_list = os.listdir(output_dir_tmp+'/train')
            print(dataset_file_list)

            n = len(dataset_file_list) // 3
            for j in range(train_file_num, train_file_num + n):
                dataset = output_dir_tmp+'/train/'+str(j-train_file_num)+'_data-feats'
                os.rename(dataset, output_dir+'/train/'+str(j)+'_data-feats')
                dataset = output_dir_tmp+'/train/'+str(j-train_file_num)+'_data-G.json'
                os.rename(dataset, output_dir+'/train/'+str(j)+'_data-G.json')
                label = output_dir_tmp+'/train/'+str(j-train_file_num)+'_label'
                os.rename(label, output_dir+'/train/'+str(j)+'_label')
            train_file_num = train_file_num + n
            
            
            cd.read_pdbbind_to_disk_rmsd_energy_split(input_list+'test', label_list_file+'/test_sb_s'+str(i)+'67.test', pdbbind_dir+str(i)+'67', groundtruth_dir,
              output_dir_tmp+'/test', resolution, tile_size, bond_th, seed = i)
            dataset_file_list = os.listdir(output_dir_tmp+'/test')
            print(dataset_file_list)

            n = len(dataset_file_list) // 3
            for j in range(test_file_num, test_file_num + n):
                dataset = output_dir_tmp+'/test/'+str(j-test_file_num)+'_data-feats'
                os.rename(dataset, output_dir+'/test/'+str(j)+'_data-feats')
                dataset = output_dir_tmp+'/test/'+str(j-test_file_num)+'_data-G.json'
                os.rename(dataset, output_dir+'/test/'+str(j)+'_data-G.json')
                label = output_dir_tmp+'/test/'+str(j-test_file_num)+'_label'
                os.rename(label, output_dir+'/test/'+str(j)+'_label')
            test_file_num = test_file_num + n
    else:
        p_list = []
        for thread_id in range(thread_num):
            p = mp.Process(target=srand_data_load_save_coord_thread, args=(input_list, label_list_file, pdbbind_dir, groundtruth_dir, output_dir, cv, resolution, bond_th, iteration, start, end, train_file_num, test_file_num, thread_num, thread_id))
            p.start()
            p_list.append(p)
    
        for p in p_list:
            p.join()


def srand_data_load_save_screen_thread(input_list, groundtruth_dir, pdbbind_dir, output_dir, resolution, bond_th, pocket_th, iteration, thread_num, thread_id):
    tile_size = 1024
    output_dir_tmp = output_dir + '_tmp_' + str(thread_id)
    if not os.path.isdir(output_dir_tmp):
        os.makedirs(output_dir_tmp)
    if not os.path.isdir(output_dir_tmp+'/train'):
        os.makedirs(output_dir_tmp+'/train')
    if not os.path.isdir(output_dir_tmp+'/test'):
        os.makedirs(output_dir_tmp+'/test')

    groundtruth_suffix = []
    if args.pdb_version == 2016:
        groundtruth_suffix = ['.rec.pdb', '.lig.mol2']
    elif args.pdb_version == 2018:
        groundtruth_suffix = ['_protein.pdb', '_ligand.mol2']

    if args.dataset == 'screen_data2':
        import screen_data2 as sd
        print("import screen_data2")
    for split in ['train', 'test']:
        input_list_filename = input_list + split
        with open(input_list_filename, 'r') as gf:
            inputs = gf.readlines()
            start = (thread_id * len(inputs)) // thread_num
            end = ((thread_id + 1) * len(inputs)) // thread_num
        sd.read_pdbbind_to_disk_rmsd_energy_split(input_list_filename,
                                                  groundtruth_dir,
                                                  groundtruth_suffix,
                                                  pdbbind_dir,
                                                  output_dir_tmp+'/'+split,
                                                  resolution,
                                                  tile_size,
                                                  bond_th,
                                                  pocket_th,
                                                  start, end)
    
    '''
    input_list_filename = input_list + 'test'
    with open(input_list_filename, 'r') as gf:
        inputs = gf.readlines()
        start = (thread_id * len(inputs)) // thread_num
        end = ((thread_id + 1) * len(inputs)) // thread_num
    sd.read_pdbbind_to_disk_rmsd_energy_split(input_list_filename,
                                              pdbbind_dir,
                                              output_dir_tmp+'/test',
                                              resolution,
                                              tile_size,
                                              bond_th,
                                              start, end)
    '''
    

def srand_data_load_save_screen(input_list, groundtruth_dir, pdbbind_dir, output_dir, resolution, bond_th, pocket_th, iteration, thread_num = 1):
    tile_size = 1024
    print("srand_data_load_save_screen")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(output_dir+'/train'):
        os.makedirs(output_dir+'/train')
    if not os.path.isdir(output_dir+'/test'):
        os.makedirs(output_dir+'/test')

    print("data dir created!")
    # for i in range(4,5):
    if thread_num == 1:
        # for i in range(start, end):
        srand_data_load_save_screen_thread(input_list, groundtruth_dir, pdbbind_dir, output_dir, resolution, bond_th, pocket_th, iteration, 1, 0)
    else:
        p_list = []
        for thread_id in range(thread_num):
            p = mp.Process(target=srand_data_load_save_screen_thread, args=(input_list, groundtruth_dir, pdbbind_dir, output_dir, resolution, bond_th, pocket_th, iteration, thread_num, thread_id))
            p.start()
            p_list.append(p)
    
        for p in p_list:
            p.join()
    for thread_id in range(thread_num):
        output_dir_tmp = output_dir + '_tmp_' + str(thread_id)
        for split in ['train', 'test']:
            # output dir of the data, with tread_id
            dataset_file_list = os.listdir(output_dir_tmp+'/'+split)
            print(dataset_file_list)
            # num of data files generated by this thread_id
            n = len(dataset_file_list) // 3
            file_num = len(os.listdir(output_dir+'/'+split)) // 3
            for j in range(file_num, file_num + n):
                dataset = output_dir_tmp+'/'+split+'/'+str(j-file_num)+'_data-feats'
                os.rename(dataset, output_dir+'/'+split+'/'+str(j)+'_data-feats')
                dataset = output_dir_tmp+'/'+split+'/'+str(j-file_num)+'_data-G.json'
                os.rename(dataset, output_dir+'/'+split+'/'+str(j)+'_data-G.json')
                label = output_dir_tmp+'/'+split+'/'+str(j-file_num)+'_label'
                os.rename(label, output_dir+'/'+split+'/'+str(j)+'_label')


def srand_data_load_save_muv_thread(input_list, groundtruth_dir, pdbbind_dir, muv_dir, target, label, output_dir, resolution, bond_th, iteration, thread_num, thread_id):
    tile_size = 1024
    output_dir_tmp = output_dir + '_tmp_' + str(thread_id)
    if not os.path.isdir(output_dir_tmp):
        os.makedirs(output_dir_tmp)
    if not os.path.isdir(output_dir_tmp+'/test'):
        os.makedirs(output_dir_tmp+'/test')

    groundtruth_suffix = []
    if args.pdb_version == 2016:
        groundtruth_suffix = ['.rec.pdb', '.lig.mol2']
    elif args.pdb_version == 2018:
        groundtruth_suffix = ['_protein.pdb', '_ligand.mol2']

    
    if args.dataset == 'muv2':
        import muv_data2 as md
        print("import muv2")
    inputs = 30 if label == 'active' else 15000
    start = (thread_id * inputs) // thread_num
    end = ((thread_id + 1) * inputs) // thread_num
    md.read_pdbbind_to_disk_rmsd_energy_split(input_list,
                                              groundtruth_dir,
                                              groundtruth_suffix,
                                              pdbbind_dir,
                                              muv_dir,
                                              target,
                                              label,
                                              output_dir_tmp+'/test',
                                              resolution,
                                              tile_size,
                                              bond_th,
                                              pocket_th,
                                              start, end)
    

def srand_data_load_save_muv(input_list, groundtruth_dir, pdbbind_dir, muv_dir, target, label, output_dir, resolution, bond_th, iteration, thread_num = 1):
    tile_size = 1024
    print("srand_data_load_save_screen")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(output_dir+'/test'):
        os.makedirs(output_dir+'/test')

    print("data dir created!")
    # for i in range(4,5):
    if thread_num == 1:
        # for i in range(start, end):
        srand_data_load_save_muv_thread(input_list, groundtruth_dir, pdbbind_dir, muv_dir, target, label, output_dir, resolution, bond_th, iteration, 1, 0)
    else:
        p_list = []
        for thread_id in range(thread_num):
            p = mp.Process(target=srand_data_load_save_muv_thread, args=(input_list, groundtruth_dir, pdbbind_dir, muv_dir, target, label, output_dir, resolution, bond_th, iteration, thread_num, thread_id))
            p.start()
            p_list.append(p)
    
        for p in p_list:
            p.join()
    for thread_id in range(thread_num):
        output_dir_tmp = output_dir + '_tmp_' + str(thread_id)
        for split in ['test']:
            # output dir of the data, with tread_id
            dataset_file_list = os.listdir(output_dir_tmp+'/'+split)
            print(dataset_file_list)
            # num of data files generated by this thread_id
            n = len(dataset_file_list) // 3
            file_num = len(os.listdir(output_dir+'/'+split)) // 3
            for j in range(file_num, file_num + n):
                dataset = output_dir_tmp+'/'+split+'/'+str(j-file_num)+'_data-feats'
                os.rename(dataset, output_dir+'/'+split+'/'+str(j)+'_data-feats')
                dataset = output_dir_tmp+'/'+split+'/'+str(j-file_num)+'_data-G.json'
                os.rename(dataset, output_dir+'/'+split+'/'+str(j)+'_data-G.json')
                label = output_dir_tmp+'/'+split+'/'+str(j-file_num)+'_label'
                os.rename(label, output_dir+'/'+split+'/'+str(j)+'_label')


def srand_data_load_save_coord_15844_thread(input_list, groundtruth_dir, pdbbind_dir, output_dir, resolution, bond_th, iteration, thread_num, thread_id):
    tile_size = 1024
    output_dir_tmp = output_dir + '_tmp_' + str(thread_id)
    if not os.path.isdir(output_dir_tmp):
        os.makedirs(output_dir_tmp)
    if not os.path.isdir(output_dir_tmp+'/train'):
        os.makedirs(output_dir_tmp+'/train')
    if not os.path.isdir(output_dir_tmp+'/test'):
        os.makedirs(output_dir_tmp+'/test')

    groundtruth_suffix = ['_protein.pdb', '_ligand.mol2']
    for split in ['train', 'test']:
        input_list_filename = input_list + split
        with open(input_list_filename, 'r') as gf:
            inputs = gf.readlines()
            start = (thread_id * len(inputs)) // thread_num
            end = ((thread_id + 1) * len(inputs)) // thread_num
        cd15844.read_pdbbind_to_disk_rmsd_energy_split(input_list_filename,
                                                       groundtruth_dir,
                                                       groundtruth_suffix,
                                                       pdbbind_dir,
                                                       output_dir_tmp+'/' + split,
                                                       resolution,
                                                       tile_size,
                                                       bond_th,
                                                       start, end)
    
    '''
    input_list_filename = input_list + 'test'
    with open(input_list_filename, 'r') as gf:
        inputs = gf.readlines()
        start = (thread_id * len(inputs)) // thread_num
        end = ((thread_id + 1) * len(inputs)) // thread_num
    cd15844.read_pdbbind_to_disk_rmsd_energy_split(input_list_filename,
                                                   groundtruth_dir,
                                                   pdbbind_dir,
                                                   output_dir_tmp+'/test',
                                                   resolution,
                                                   tile_size,
                                                   bond_th,
                                                   start, end)
    '''
    

def srand_data_load_save_coord_15844(input_list, groundtruth_dir, pdbbind_dir, output_dir, resolution, bond_th, iteration, thread_num = 1):
    tile_size = 1024
    print("srand_data_load_save_coord_15844")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(output_dir+'/train'):
        os.makedirs(output_dir+'/train')
    if not os.path.isdir(output_dir+'/test'):
        os.makedirs(output_dir+'/test')

    print("data dir created!")
    # for i in range(4,5):
    if thread_num == 1:
        # for i in range(start, end):
        srand_data_load_save_coord_15844_thread(input_list, groundtruth_dir, pdbbind_dir, output_dir, resolution, bond_th, iteration, 1, 0)
    else:
        p_list = []
        for thread_id in range(thread_num):
            p = mp.Process(target=srand_data_load_save_coord_15844_thread,
                           args=(input_list, groundtruth_dir, pdbbind_dir, output_dir, resolution, bond_th, iteration, thread_num, thread_id))
            p.start()
            p_list.append(p)
    
        for p in p_list:
            p.join()
    for thread_id in range(thread_num):
        output_dir_tmp = output_dir + '_tmp_' + str(thread_id)
        # for split in ['test']:
        for split in ['train', 'test']:
            # output dir of the data, with tread_id
            dataset_file_list = os.listdir(output_dir_tmp+'/'+split)
            print(dataset_file_list)
            # num of data files generated by this thread_id
            n = len(dataset_file_list) // 3
            file_num = len(os.listdir(output_dir+'/'+split)) // 3
            for j in range(file_num, file_num + n):
                dataset = output_dir_tmp+'/'+split+'/'+str(j-file_num)+'_data-feats'
                os.rename(dataset, output_dir+'/'+split+'/'+str(j)+'_data-feats')
                dataset = output_dir_tmp+'/'+split+'/'+str(j-file_num)+'_data-G.json'
                os.rename(dataset, output_dir+'/'+split+'/'+str(j)+'_data-G.json')
                label = output_dir_tmp+'/'+split+'/'+str(j-file_num)+'_label'
                os.rename(label, output_dir+'/'+split+'/'+str(j)+'_label')


def srand_data_load_save_coord2_thread(input_list, groundtruth_dir, pdbbind_dir, output_dir, resolution, bond_th, pocket_th, iteration, thread_num, thread_id):
    tile_size = 1024
    output_dir_tmp = output_dir + '_tmp_' + str(thread_id)
    if not os.path.isdir(output_dir_tmp):
        os.makedirs(output_dir_tmp)
    if not os.path.isdir(output_dir_tmp+'/train'):
        os.makedirs(output_dir_tmp+'/train')
    if not os.path.isdir(output_dir_tmp+'/test'):
        os.makedirs(output_dir_tmp+'/test')

    groundtruth_suffix = []
    if args.pdb_version == 2016:
        groundtruth_suffix = ['.rec.pdb', '.lig.mol2']
    elif args.pdb_version == 2018:
        groundtruth_suffix = ['_protein.pdb', '_ligand.mol2']
    elif args.pdb_version == 2022:
        groundtruth_suffix = ['_protein.pdb', '_ligand_opt.mol2']

    if args.dataset == 'autodock':
        import autodock_data as cd2
        splits = ['test']
        print("import autodock_data")
    elif args.dataset == 'coor2':
        import coordinate_data2 as cd2
        splits = ['train', 'test']
        print("import coordinate_data2")

    for split in splits:
        input_list_filename = input_list + split
        with open(input_list_filename, 'r') as gf:
            inputs = gf.readlines()
            start = (thread_id * len(inputs)) // thread_num
            end = ((thread_id + 1) * len(inputs)) // thread_num
        cd2.read_pdbbind_to_disk_rmsd_energy_split(input_list_filename,
                                                   groundtruth_dir,
                                                   groundtruth_suffix,
                                                   pdbbind_dir,
                                                   output_dir_tmp+'/' + split,
                                                   resolution,
                                                   tile_size,
                                                   bond_th,
                                                   pocket_th,
                                                   start, end)
    

def srand_data_load_save_coord2(input_list, groundtruth_dir, pdbbind_dir, output_dir, resolution, bond_th, pocket_th, iteration, thread_num = 1):
    tile_size = 1024
    print("srand_data_load_save_coord2")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(output_dir+'/train'):
        os.makedirs(output_dir+'/train')
    if not os.path.isdir(output_dir+'/test'):
        os.makedirs(output_dir+'/test')

    print("data dir created!")
    # for i in range(4,5):
    if thread_num == 1:
        # for i in range(start, end):
        srand_data_load_save_coord2_thread(input_list, groundtruth_dir, pdbbind_dir, output_dir, resolution, bond_th, pocket_th, iteration, 1, 0)
    else:
        p_list = []
        for thread_id in range(thread_num):
            p = mp.Process(target=srand_data_load_save_coord2_thread,
                           args=(input_list, groundtruth_dir, pdbbind_dir, output_dir, resolution, bond_th, pocket_th, iteration, thread_num, thread_id))
            p.start()
            p_list.append(p)
    
        for p in p_list:
            p.join()
    for thread_id in range(thread_num):
        output_dir_tmp = output_dir + '_tmp_' + str(thread_id)
        # for split in ['test']:
        for split in ['train', 'test']:
            # output dir of the data, with tread_id
            dataset_file_list = os.listdir(output_dir_tmp+'/'+split)
            print(dataset_file_list)
            # num of data files generated by this thread_id
            n = len(dataset_file_list) // 3
            file_num = len(os.listdir(output_dir+'/'+split)) // 3
            for j in range(file_num, file_num + n):
                dataset = output_dir_tmp+'/'+split+'/'+str(j-file_num)+'_data-feats'
                os.rename(dataset, output_dir+'/'+split+'/'+str(j)+'_data-feats')
                dataset = output_dir_tmp+'/'+split+'/'+str(j-file_num)+'_data-G.json'
                os.rename(dataset, output_dir+'/'+split+'/'+str(j)+'_data-G.json')
                label = output_dir_tmp+'/'+split+'/'+str(j-file_num)+'_label'
                os.rename(label, output_dir+'/'+split+'/'+str(j)+'_label')


def srand_data_load_save_casf_thread(input_list, groundtruth_dir, pdbbind_dir, casf_groundtruth_dir, casf_dir, output_dir,
                                     resolution, bond_th, pocket_th, iteration, thread_num, thread_id):
    tile_size = 1024
    output_dir_tmp = output_dir + '_tmp_' + str(thread_id)
    if not os.path.isdir(output_dir_tmp):
        os.makedirs(output_dir_tmp)
    if not os.path.isdir(output_dir_tmp+'/train'):
        os.makedirs(output_dir_tmp+'/train')
    if not os.path.isdir(output_dir_tmp+'/test'):
        os.makedirs(output_dir_tmp+'/test')

    groundtruth_suffix = {}
    if args.pdb_version == 2016:
        groundtruth_suffix['train'] = ['.rec.pdb', '.lig.mol2']
    elif args.pdb_version == 2018:
        groundtruth_suffix['train'] = ['_protein.pdb', '_ligand.mol2']
    groundtruth_suffix['test'] = ['_protein.pdb', '_ligand_opt.mol2']

    data_dir = {'train': pdbbind_dir, 'test': casf_dir}
    GT_dir = {'train': groundtruth_dir, 'test': casf_groundtruth_dir}

    import coordinate_data2 as cd2
    # import casf_data as cad

    func = {'train': cd2, 'test': cd2}
    splits = ['train', 'test']

    for split in splits:
        input_list_filename = input_list + split
        with open(input_list_filename, 'r') as gf:
            inputs = gf.readlines()
            start = (thread_id * len(inputs)) // thread_num
            end = ((thread_id + 1) * len(inputs)) // thread_num
        func[split].read_pdbbind_to_disk_rmsd_energy_split(input_list_filename,
                                                           GT_dir[split],
                                                           groundtruth_suffix[split],
                                                           data_dir[split],
                                                           output_dir_tmp+'/' + split,
                                                           resolution,
                                                           tile_size,
                                                           bond_th,
                                                           pocket_th,
                                                           start, end)
    

def srand_data_load_save_casf(input_list, groundtruth_dir, pdbbind_dir, casf_groundtruth_dir, casf_dir, output_dir,
                              resolution, bond_th, pocket_th, iteration, thread_num = 1):
    tile_size = 1024
    print("srand_data_load_save_coord2")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(output_dir+'/train'):
        os.makedirs(output_dir+'/train')
    if not os.path.isdir(output_dir+'/test'):
        os.makedirs(output_dir+'/test')

    print("data dir created!")
    # for i in range(4,5):
    if thread_num == 1:
        # for i in range(start, end):
        srand_data_load_save_casf_thread(input_list, groundtruth_dir, pdbbind_dir, casf_groundtruth_dir, casf_dir, output_dir, resolution, bond_th, pocket_th, iteration, 1, 0)
    else:
        p_list = []
        for thread_id in range(thread_num):
            p = mp.Process(target=srand_data_load_save_casf_thread,
                           args=(input_list, groundtruth_dir, pdbbind_dir, casf_groundtruth_dir, casf_dir, output_dir,
                                 resolution, bond_th, pocket_th, iteration, thread_num, thread_id))
            p.start()
            p_list.append(p)
    
        for p in p_list:
            p.join()
    for thread_id in range(thread_num):
        output_dir_tmp = output_dir + '_tmp_' + str(thread_id)
        # for split in ['test']:
        for split in ['train', 'test']:
            # output dir of the data, with tread_id
            dataset_file_list = os.listdir(output_dir_tmp+'/'+split)
            print(dataset_file_list)
            # num of data files generated by this thread_id
            n = len(dataset_file_list) // 3
            file_num = len(os.listdir(output_dir+'/'+split)) // 3
            for j in range(file_num, file_num + n):
                dataset = output_dir_tmp+'/'+split+'/'+str(j-file_num)+'_data-feats'
                os.rename(dataset, output_dir+'/'+split+'/'+str(j)+'_data-feats')
                dataset = output_dir_tmp+'/'+split+'/'+str(j-file_num)+'_data-G.json'
                os.rename(dataset, output_dir+'/'+split+'/'+str(j)+'_data-G.json')
                label = output_dir_tmp+'/'+split+'/'+str(j-file_num)+'_label'
                os.rename(label, output_dir+'/'+split+'/'+str(j)+'_label')


def split_srand_label(cv, input_list, label_list_file):

    for i in range(3,14):
        f = open('/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/check_rmsd/sb_s' + str(i) + '67.test', 'r')
        # f_train = open('/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/check_rmsd/label_train/sb_s' + str(i) + '67.test', 'w')
        # f_test = open('/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/check_rmsd/label_test/sb_s' + str(i) + '67.test', 'w')
        
        # f = open('/home/mdl/hzj5142/AtomNet/cross_validation/check_rmsd/sb_s' + str(i) + '67.test', 'r')
        # f_train = open('/home/mdl/hzj5142/AtomNet/cross_validation/cv' + str(cv) + '/train_sb_s' + str(i) + '67.test', 'w')
        # f_test = open('/home/mdl/hzj5142/AtomNet/cross_validation/cv' + str(cv) + '/test_sb_s' + str(i) + '67.test', 'w')
        f_train = open(label_list_file + '/train_sb_s' + str(i) + '67.test', 'w')
        f_test = open(label_list_file + '/test_sb_s' + str(i) + '67.test', 'w')



        rec_list = []
        # ff = open('/home/mdl/hzj5142/AtomNet/cross_validation/pdbbind_3875_list', 'r')
        ff = open('/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/pdbbind_3875_list', 'r')
        for line in ff:
            # rec_list.append(line.strip()+'.pdb')
            rec_list.append(line.strip())
        ff.close()

        rmsd_list = []
        for j in range(3875):
            line = f.readline()
            st = line.split()
            rmsd_list.append(st)

        ff = open(input_list+'train', 'r')
        for line in ff:
            protein = line.strip()
            index = rec_list.index(protein)
            st = rmsd_list[index]
            for sss in st:
                f_train.write(sss+'\n')
        ff.close()

        ff = open(input_list+'test', 'r')
        for line in ff:
            protein = line.strip()
            index = rec_list.index(protein)
            st = rmsd_list[index]
            for sss in st:
                f_test.write(sss+'\n')
        ff.close()




        f.close()
        f_train.close()
        f_test.close()


if __name__ == "__main__":
    #input_list = sys.argv[1]
    #label_list_file = sys.argv[2]
    #pdbbind_dir = sys.argv[3]
    #groundtruth_dir = sys.argv[4]
    #output_dir = sys.argv[5]
    #resolution = int(sys.argv[6])

    # cv = int(sys.argv[1])
    cv = args.cv
    # input_list = '/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/AtomNet/tmp_data/pdbbind_'
    #input_list = '/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/medusa/run_medusa/pdbbind_'
    # label_list_file = '/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/check_rmsd/label_'
    # label_list_file = '/gpfs/group/mtk2/cyberstar/hzj5142/GNN/GNN/DGNN/data/pdbbind' # + str(cv) #+ '/train_sb_s'
    label_list_file = args.label_list_file

    
    # pdbbind_dir = '/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/medusa/pdbbind_output_s'
    pdbbind_dir = args.pdbbind_dir
    # pdbbind_dir = '/home/mdl/hzj5142/AtomNet/medusa/pdbbind_output_s'
    
    groundtruth_dir = '/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/pdbbind/'
    groundtruth_dir = args.groundtruth_dir
    # groundtruth_dir = '/home/mdl/hzj5142/AtomNet/medusa/pdbbind'
    #output_dir = '/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/AtomNet/tmp_data/pdbbind_rmsd_srand2_30'
    #resolution = 30

    # iteration = int(sys.argv[2])
    iteration = args.resolution
    #iteration = 0
    resolution = iteration
    bond_th = args.bond_th
    pocket_th = args.pocket_th
    #output_dir = '/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/AtomNet/tmp_data/pdbbind_rmsd_srand2_acc_'+str(resolution)
    # cv = int(sys.argv[1])

    # output_dir = '/home/mdl/hzj5142/AtomNet/cross_validation/cv' + str(cv) + '/pdbbind_rmsd_srand'
    # input_list = '/home/mdl/hzj5142/AtomNet/cross_validation/cv' + str(cv) + '/pdb_list_'
    # output_file = sys.argv[3]
    output_file = args.output_file
    output_dir = label_list_file + '/' + output_file
    # output_dir = label_list_file + '/pdbbind_rmsd_srand2'
    input_list = label_list_file + '/pdb_list_'
    input_list = args.input_list

    # split_srand_label(cv, input_list, label_list_file)

    # srand_data_load_save(input_list, label_list_file, pdbbind_dir, groundtruth_dir, output_dir, cv, resolution, iteration)
    casf_groundtruth_dir = args.casf_groundtruth_dir
    casf_dir = args.casf_dir

    # start = int(sys.argv[4])
    # end = int(sys.argv[5])
    start = args.start_iter
    end = args.end_iter
    # thread_num = int(sys.argv[6])
    thread_num = args.thread_num
    use_new = args.use_new_data
    screen_data = args.screen_data
    print(cv, iteration, output_file, start, end, thread_num, use_new)
    if use_new:
        if args.dataset in ['screen_data', 'screen_data2']:
            srand_data_load_save_screen(input_list, groundtruth_dir, pdbbind_dir, output_dir, resolution, bond_th, pocket_th, iteration, thread_num = thread_num)
        elif args.dataset == 'coor':
            srand_data_load_save_coord(input_list, label_list_file, pdbbind_dir, groundtruth_dir, output_dir, cv, resolution, bond_th, iteration, start, end, thread_num = thread_num)
        elif args.dataset == 'coor_15844':
            srand_data_load_save_coord_15844(input_list, groundtruth_dir, pdbbind_dir, output_dir, resolution, bond_th, iteration, thread_num = thread_num)
        elif args.dataset in ['muv', 'muv2']:
            srand_data_load_save_muv(input_list, groundtruth_dir, pdbbind_dir, args.muv_dir, args.muv_target, args.muv_label, output_dir, resolution, bond_th, iteration, thread_num = thread_num)
        elif args.dataset in ['coor2', 'autodock']:
            srand_data_load_save_coord2(input_list, groundtruth_dir, pdbbind_dir, output_dir, resolution, bond_th, pocket_th, iteration, thread_num = thread_num)
        elif args.dataset in ['casf']:
            srand_data_load_save_casf(input_list, groundtruth_dir, pdbbind_dir, casf_groundtruth_dir, casf_dir, output_dir,
                                      resolution, bond_th, pocket_th, iteration, thread_num = thread_num)
    else:
        srand_data_load_save_gcn(input_list, label_list_file, pdbbind_dir, groundtruth_dir, output_dir, cv, resolution, iteration, start, end, thread_num = thread_num)

    # load_and_save_dataset(input_list, label_list_file, pdbbind_dir, groundtruth_dir, output_dir, resolution)

