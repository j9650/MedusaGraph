# MedusaGraph

# To run MedusaGraph, please use the following command line (for 1nja)
python medusagraph.py --ligand_file=data/pdbbind/1nja/1nja.lig.mol2 --protein_file=data/pdbbind/1nja/1nja.rec.pdb --pose_file=data/medusadock_output/1nja.pdb --output_file=MedusaGraph_tmp/1nja.mol2 --prediction_model=models/prediction_model.pt --selection_model=models/selection_model.pt --tmp_dir=MedusaGraph_tmp/

# To train the model by your self, you need to first generated the dataset:
python convert_data_to_disk.py --cv=0 --input_list=data/pdb_list_ --output_file=pdbbind_rmsd_srand_coor2 --thread_num=1 --use_new_data --bond_th=6 --pocket_th=12 --groundtruth_dir=data/pdbbind/ --pdbbind_dir=data/medusadock_output --label_list_file=MedusaGraph_tmp --dataset=coor2 --pdb_version=2016

mkdir MedusaGraph_tmp/pdbbind_rmsd_srand_coor2/raw

mv MedusaGraph_tmp/pdbbind_rmsd_srand_coor2/test/ MedusaGraph_tmp/pdbbind_rmsd_srand_coor2/raw/

mv MedusaGraph_tmp/pdbbind_rmsd_srand_coor2/train/ MedusaGraph_tmp/pdbbind_rmsd_srand_coor2/raw/
