# python convert_data_to_disk.py /gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/AtomNet/tmp_data/pdbbind_ /gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/AtomNet/tmp_data/pdbbind_ /gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/medusa/pdbbind_output/ /gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/pdbbind/ /gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/AtomNet/tmp_data/pdbbind_rmsd_resolution30 30

#import tensorflow as tf
import numpy as np
import os
import sys
#from sklearn import metrics

import multiprocessing as mp

import dataset_from_file as dff
import gcn_data as gd

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

    cv = int(sys.argv[1])
    # input_list = '/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/AtomNet/tmp_data/pdbbind_'
    #input_list = '/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/medusa/run_medusa/pdbbind_'
    # label_list_file = '/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/check_rmsd/label_'
    label_list_file = '/gpfs/group/mtk2/cyberstar/hzj5142/GNN/GNN/DGNN/data/pdbbind' # + str(cv) #+ '/train_sb_s'
    
    pdbbind_dir = '/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/medusa/pdbbind_output_s'
    # pdbbind_dir = '/home/mdl/hzj5142/AtomNet/medusa/pdbbind_output_s'
    
    groundtruth_dir = '/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/pdbbind/'
    # groundtruth_dir = '/home/mdl/hzj5142/AtomNet/medusa/pdbbind'
    #output_dir = '/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/AtomNet/tmp_data/pdbbind_rmsd_srand2_30'
    #resolution = 30

    iteration = int(sys.argv[2])
    #iteration = 0
    resolution = iteration
    #output_dir = '/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/AtomNet/tmp_data/pdbbind_rmsd_srand2_acc_'+str(resolution)
    # cv = int(sys.argv[1])

    # output_dir = '/home/mdl/hzj5142/AtomNet/cross_validation/cv' + str(cv) + '/pdbbind_rmsd_srand'
    # input_list = '/home/mdl/hzj5142/AtomNet/cross_validation/cv' + str(cv) + '/pdb_list_'
    output_file = sys.argv[3]
    output_dir = label_list_file + '/' + output_file
    # output_dir = label_list_file + '/pdbbind_rmsd_srand2'
    input_list = label_list_file + '/pdb_list_'

    # split_srand_label(cv, input_list, label_list_file)

    # srand_data_load_save(input_list, label_list_file, pdbbind_dir, groundtruth_dir, output_dir, cv, resolution, iteration)

    start = int(sys.argv[4])
    end = int(sys.argv[5])
    thread_num = int(sys.argv[6])
    srand_data_load_save_gcn(input_list, label_list_file, pdbbind_dir, groundtruth_dir, output_dir, cv, resolution, iteration, start, end, thread_num = thread_num)

    # load_and_save_dataset(input_list, label_list_file, pdbbind_dir, groundtruth_dir, output_dir, resolution)

