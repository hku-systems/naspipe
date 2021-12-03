# Artifact Evaluation

[![DOI](https://zenodo.org/badge/433303490.svg)](https://zenodo.org/badge/latestdoi/433303490)


Artifact Ver 0.1 - Kick Off Version 

## Dataset and AE hosts

You can download the dataset via this link: https://drive.google.com/file/d/1QztbVol4kaQjIL3lpuGvLkBRTB_6Fd1-/view?usp=sharing

You can unzip it to naspipe-ae/[experiment name]/translation/data.

If reviewers need a bare metal host to evaluate our artifact, please leave a message in the hotcrp.

## Installation 

Pull and run PyTorch official image:
```bash
docker pull pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel
```
```bash
cd naspipe-ae/
nvidia-docker run -it -v $PWD:/workspace --net=host --ipc=host --name=naspipe pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel
```
Inside docker:
```bash
apt-get update
apt-get install -y git
cd /
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

Fix fairseq compatibility issues: 

```bash
git am < /workspace/0001-compatible.patch
cd /workspace
```

Enable deterministic execution:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## Kick-off Functional

```bash
cd /workspace/reproducible/translation
```
Run sequential execution (no parallel):

```bash
python -m launch --nnodes 1 --node_rank 0 --nproc_per_node 4 main_with_runtime_single.py --data_dir data/wmt14_en_de_joined_dict --master_addr localhost --module gpus=4 --checkpoint_dir output --distributed_backend gloo -b 3840 --lr 0.000060 --lr_policy polynomial --weight-decay 0.000000 --epochs 10 --print-freq 10 --verbose 0 --num_ranks_in_server 4 --config_path gpus=4/mp_conf.json
```

If you see the the following logs, the artifact runs perfectly.

```bash
Stage: [3] Epoch: [0][1/49372]	Time: 0.784 (0.784)	Id: 0	Tokens: 761	Output: 8336.44824218750000000000000000000000	
Stage: [3] Epoch: [0][2/49372]	Time: 1.943 (1.364)	Id: 1	Tokens: 3293	Output: 36547.63671875000000000000000000000000	
Stage: [3] Epoch: [0][3/49372]	Time: 1.803 (1.510)	Id: 2	Tokens: 3136	Output: 34344.91406250000000000000000000000000	
Stage: [3] Epoch: [0][4/49372]	Time: 1.467 (1.499)	Id: 3	Tokens: 1717	Output: 18729.03515625000000000000000000000000	
Stage: [3] Epoch: [0][5/49372]  Time: 2.070 (1.613) Id: 4   Tokens: 3520    Output: 38475.35156250000000000000000000000000
```


## Functional Experiment

(to be released soon within the kick-off period)

Experiment 1:

./run_compare.sh


Experiment 2:

./run_throughput.sh
