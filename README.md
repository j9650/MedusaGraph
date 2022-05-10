# MedusaGraph

# To run MedusaGraph, please use the following command line (e.g., for 1nja)
python medusagraph.py --ligand_file=data/pdbbind/1nja/1nja.lig.mol2 --protein_file=data/pdbbind/1nja/1nja.rec.pdb --pose_file=data/medusadock_output/1nja.pdb --output_file=MedusaGraph_tmp/1nja.mol2 --prediction_model=models/prediction_model.pt --selection_model=models/selection_model.pt --tmp_dir=MedusaGraph_tmp/

# To train the model by your self, you need to first generated the dataset:
python convert_data_to_disk.py --cv=0 --input_list=data/pdb_list_ --output_file=pdbbind_rmsd_srand_coor2 --thread_num=1 --use_new_data --bond_th=6 --pocket_th=12 --groundtruth_dir=data/pdbbind/ --pdbbind_dir=data/medusadock_output --label_list_file=MedusaGraph_tmp --dataset=coor2 --pdb_version=2016

mkdir MedusaGraph_tmp/pdbbind_rmsd_srand_coor2/raw

mv MedusaGraph_tmp/pdbbind_rmsd_srand_coor2/test/ MedusaGraph_tmp/pdbbind_rmsd_srand_coor2/raw/

mv MedusaGraph_tmp/pdbbind_rmsd_srand_coor2/train/ MedusaGraph_tmp/pdbbind_rmsd_srand_coor2/raw/

# Then train the pose prediction model with:
python train_coor.py --gpu_id=0 --n_graph_layer=4 --d_graph_layer=256 --start_epoch=1 --epoch=100 --flexible --model_dir=MedusaGraph_tmp/models_4_256_atom_hinge0 --data_path=MedusaGraph_tmp/pdbbind_rmsd_srand_coor2 --heads=1 --batch_size=1 --model_type=Net_coor --residue --edge_dim=3 --loss=L1Loss --loss_reduction=mean --output=MedusaGraph_tmp/output_4_256_atom_hinge0 --hinge=0 --tot_seed=1

# Prepaire the data for training the pose selection model:
python test.py --path=MedusaGraph_tmp/pdbbind_rmsd_srand_coor2_2 --pre_root=MedusaGraph_tmp/pdbbind_rmssrand_coor2 --model_dir=models/prediction_model.pt --dataset=PDBBindNextStep2 --gpu_id=0

# Train the pose selection model:
python train_select.py --model_type=Net_screen --epoch=50 --gpu_id=0 --data_path=MedusaGraph_tmp/pdbbind_rmsd_srand_coor2_2/ --n_graph_layer=2 --d_graph_layer=256 --n_FC_layer=2 --d_FC_layer=256 --output=MedusaGraph_tmp/models_4_256_atom_hinge0_2/out --model_dir=MedusaGraph_tmp/models_4_256_atom_hinge0_2 --loss=CrossEntropyLoss --loss_reduction=mean --flexible --weight_bias=-9.0 --th=0.03
