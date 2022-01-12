#!/bin/bash

cd /workspace/throughput/translation
export PYTHONWARNINGS="ignore"
echo "Experiment starts..."
# naspipe 
echo "NasPipe."
# c0
echo "c0."
python -m autolaunch --collectsteps 200 --nnodes 1 --node_rank 0 --nproc_per_node 4 main_with_runtime.py \
--data_dir data/wmt14_en_de_joined_dict --master_addr localhost --module gpus=4 \
--checkpoint_dir output --distributed_backend gloo -b 3840 --lr 0.000060 \
--lr_policy polynomial --weight-decay 0.000000 --epochs 10 --print-freq 10 \
--verbose 0 --num_ranks_in_server 4 --config_path gpus=4/mp_conf.json \
--input_path config_4_4.json

# c1
echo "c1."
python -m autolaunch --collectsteps 200 --nnodes 1 --node_rank 0 --nproc_per_node 4 main_with_runtime.py \
--data_dir data/wmt14_en_de_joined_dict --master_addr localhost --module gpus=4 \
--checkpoint_dir output --distributed_backend gloo -b 3840 --lr 0.000060 \
--lr_policy polynomial --weight-decay 0.000000 --epochs 10 --print-freq 10 \
--verbose 0 --num_ranks_in_server 4 --config_path gpus=4/mp_conf.json \
--input_path config_4_3.json

echo "c2."
# c2
python -m autolaunch --collectsteps 200 --nnodes 1 --node_rank 0 --nproc_per_node 4 main_with_runtime.py \
--data_dir data/wmt14_en_de_joined_dict --master_addr localhost --module gpus=4 \
--checkpoint_dir output --distributed_backend gloo -b 3840 --lr 0.000060 \
--lr_policy polynomial --weight-decay 0.000000 --epochs 10 --print-freq 10 \
--verbose 0 --num_ranks_in_server 4 --config_path gpus=4/mp_conf.json \
--input_path config_4_2.json

echo "c3."
# c3 
python -m autolaunch --collectsteps 200 --nnodes 1 --node_rank 0 --nproc_per_node 4 main_with_runtime.py \
--data_dir data/wmt14_en_de_joined_dict --master_addr localhost --module gpus=4 \
--checkpoint_dir output --distributed_backend gloo -b 3840 --lr 0.000060 \
--lr_policy polynomial --weight-decay 0.000000 --epochs 10 --print-freq 10 \
--verbose 0 --num_ranks_in_server 4 --config_path gpus=4/mp_conf.json \
--input_path config_4.json


cd /workspace/baselines/translation


echo "VPipe."

echo "c0."
python -m autolaunch --collectsteps 200 --nnodes 1 --node_rank 0 --nproc_per_node 4 main_with_runtime.py \
--data_dir data/wmt14_en_de_joined_dict --master_addr localhost --module vgpus=4 \
--checkpoint_dir output --distributed_backend gloo  --lr 0.000060 \
--lr_policy polynomial --weight-decay 0.000000 --epochs 10 --rep 16 --print-freq 10 \
--verbose 0 --num_ranks_in_server 4 --config_path vgpus=4/mp_conf.json --num_minibatches 5000 \
--batch_size 192 --input_path=config_4_3.json --sys=vpipe

echo "c1."
python -m autolaunch --collectsteps 200 --nnodes 1 --node_rank 0 --nproc_per_node 4 main_with_runtime.py \
--data_dir data/wmt14_en_de_joined_dict --master_addr localhost --module vgpus=4 \
--checkpoint_dir output --distributed_backend gloo  --lr 0.000060 \
--lr_policy polynomial --weight-decay 0.000000 --epochs 10 --rep 16 --print-freq 10 \
--verbose 0 --num_ranks_in_server 4 --config_path vgpus=4/mp_conf.json --num_minibatches 5000 \
--batch_size 192 --input_path=config_4_3.json --sys=vpipe


echo "c2."
python -m autolaunch --collectsteps 200 --nnodes 1 --node_rank 0 --nproc_per_node 4 main_with_runtime.py \
--data_dir data/wmt14_en_de_joined_dict --master_addr localhost --module vgpus=4 \
--checkpoint_dir output --distributed_backend gloo  --lr 0.000060 \
--lr_policy polynomial --weight-decay 0.000000 --epochs 10 --rep 16 --print-freq 10 \
--verbose 0 --num_ranks_in_server 4 --config_path vgpus=4/mp_conf.json --num_minibatches 5000 \
--batch_size 192 --input_path=config_4_2.json --sys=vpipe


echo "c3."
python -m autolaunch --collectsteps 200 --nnodes 1 --node_rank 0 --nproc_per_node 4 main_with_runtime.py \
--data_dir data/wmt14_en_de_joined_dict --master_addr localhost --module vgpus=4 \
--checkpoint_dir output --distributed_backend gloo  --lr 0.000060 \
--lr_policy polynomial --weight-decay 0.000000 --epochs 10 --rep 16 --print-freq 10 \
--verbose 0 --num_ranks_in_server 4 --config_path vgpus=4/mp_conf.json --num_minibatches 5000 \
--batch_size 192 --input_path=config_4.json --sys=vpipe


echo "GPipe."

echo "c1."
python -m autolaunch --collectsteps 1200 --nnodes 1 --node_rank 0 --nproc_per_node 4 main_with_runtime.py \
--data_dir data/wmt14_en_de_joined_dict --master_addr localhost --module gpus=4 \
--checkpoint_dir output --distributed_backend gloo  --lr 0.000060 \
--lr_policy polynomial --weight-decay 0.000000 --epochs 10 --print-freq 10 \
--verbose 0 --num_ranks_in_server 4 --config_path gpus=4/mp_conf.json --num_minibatches 5000 \
--batch_size 32 --input_path=config_4_3.json --sys=gpipe


echo "c2."
python -m autolaunch --collectsteps 600 --nnodes 1 --node_rank 0 --nproc_per_node 4 main_with_runtime.py \
--data_dir data/wmt14_en_de_joined_dict --master_addr localhost --module gpus=4 \
--checkpoint_dir output --distributed_backend gloo  --lr 0.000060 \
--lr_policy polynomial --weight-decay 0.000000 --epochs 10 --print-freq 10 \
--verbose 0 --num_ranks_in_server 4 --config_path gpus=4/mp_conf.json --num_minibatches 5000 \
--batch_size 64 --input_path=config_4_2.json --sys=gpipe


echo "c3."
python -m autolaunch --collectsteps 200 --nnodes 1 --node_rank 0 --nproc_per_node 4 main_with_runtime.py \
--data_dir data/wmt14_en_de_joined_dict --master_addr localhost --module gpus=4 \
--checkpoint_dir output --distributed_backend gloo  --lr 0.000060 \
--lr_policy polynomial --weight-decay 0.000000 --epochs 10 --print-freq 10 \
--verbose 0 --num_ranks_in_server 4 --config_path gpus=4/mp_conf.json --num_minibatches 5000 \
--batch_size 128 --input_path=config_4.json --sys=gpipe

echo "Pipedream."

echo "c1."
python -m autolaunch --collectsteps 1200 --nnodes 1 --node_rank 0 --nproc_per_node 4 main_with_runtime.py \
--data_dir data/wmt14_en_de_joined_dict --master_addr localhost --module pgpus=4 \
--checkpoint_dir output --distributed_backend gloo  --lr 0.000060 \
--lr_policy polynomial --weight-decay 0.000000 --epochs 10 --print-freq 10 \
--verbose 0 --num_ranks_in_server 4 --config_path pgpus=4/mp_conf.json --num_minibatches 5000 \
--batch_size 16 --input_path=config_4_3.json --sys=pipedream


echo "c2."
python -m autolaunch --collectsteps 1200 --nnodes 1 --node_rank 0 --nproc_per_node 4 main_with_runtime.py \
--data_dir data/wmt14_en_de_joined_dict --master_addr localhost --module pgpus=4 \
--checkpoint_dir output --distributed_backend gloo  --lr 0.000060 \
--lr_policy polynomial --weight-decay 0.000000 --epochs 10 --print-freq 10 \
--verbose 0 --num_ranks_in_server 4 --config_path pgpus=4/mp_conf.json --num_minibatches 5000 \
--batch_size 24 --input_path=config_4_2.json --sys=pipedream


echo "c3."
python -m autolaunch --collectsteps 600 --nnodes 1 --node_rank 0 --nproc_per_node 4 main_with_runtime.py \
--data_dir data/wmt14_en_de_joined_dict --master_addr localhost --module pgpus=4 \
--checkpoint_dir output --distributed_backend gloo  --lr 0.000060 \
--lr_policy polynomial --weight-decay 0.000000 --epochs 10 --print-freq 10 \
--verbose 0 --num_ranks_in_server 4 --config_path pgpus=4/mp_conf.json --num_minibatches 5000 \
--batch_size 48 --input_path=config_4.json --sys=pipedream







