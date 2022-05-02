# MedusaGraph

# command line to generate train/test data
python convert_data_to_disk.py 0 12 pdbbind_rmsd_srand13 3 14 11
# the processed data will be stored at $output_dir/pdbbind_rmsd_srand13
# Here, $output_dir is set in convert_data_to_disk.py



# the script to train MedusaGraph, change the parameters in test_train.sh
sh test_train.sh
