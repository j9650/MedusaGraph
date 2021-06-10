bd=_
train=train
gpu=3
ngraphlayer=2
dgraphlayer=1024
nFClayer=2
dFClayer=1024
batchsize=50
epoch=100
start=0
dr=0.5
ddr=05
datanum=13
weightbias=1.0
wb=10
last=log
KD=No
KD_soft=exp

jobname=edge_weight_f_data$datanum$bd$train$bd$ngraphlayer$bd$nFClayer$bd$batchsize$bd$epoch$bd$ddr$bd$wb$last$KD$KD_soft
datapath=/home/mdl/hzj5142/GNN/pdb-gnn/data/pdbbind_rmsd_srand$datanum
modeldir=models/$jobname
output=train_output/$jobname.out

python train.py --gpu_id $gpu --weight_bias $weightbias --n_graph_layer $ngraphlayer --d_graph_layer $dgraphlayer --n_FC_layer $nFClayer --d_FC_layer $dFClayer --batch_size $batchsize --epoch $epoch --dropout_rate $dr --last $last --data_path $datapath --model_dir $modeldir --output $output --KD $KD --KD_soft $KD_soft --start_epoch $start --edge False
