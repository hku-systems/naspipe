# Artifact Evaluation

[![DOI](https://zenodo.org/badge/433303490.svg)](https://zenodo.org/badge/latestdoi/433303490)


Artifact Ver 0.1 - Kick Off Version 


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

### Experiment 1:

./run_compare.sh


### Experiment 2:

In Experiment 2, we provide the original environment where we conducted our performance evaluation.

Pull the transformer image with apex installed:

```bash
docker pull zsxhku/transformer:apex
```

Start docker under the AE directory:
```bash
cd naspipe-ae/
nvidia-docker run -it -v $PWD:/workspace --net=host --ipc=host zsxhku/transformer:apex
cd /workspace/throughput/translation
```

Run throughput with different search configurations by modifying the last input argument (e.g., modifiy --input_path [space_config] to --input_path config_4_4.json): 

```bash
python -m launch --nnodes 1 --node_rank 0 --nproc_per_node 4 main_with_runtime.py --data_dir data/wmt14_en_de_joined_dict --master_addr localhost --module gpus=4 --checkpoint_dir output --distributed_backend gloo -b 3840 --lr 0.000060 --lr_policy polynomial --weight-decay 0.000000 --epochs 10 --print-freq 10 --verbose 0 --num_ranks_in_server 4 --config_path gpus=4/mp_conf.json --input_path [space_config]
```

#### space_config c0 file: config_4_4.json

Expected output at the 200 step, estimated time to execute 1K subnets is (0.129):

```bash
Stage: [3] Epoch: [0][200/1000] Time(1639706538.722864): 0.329 (0.404)  Epoch time [hr]: 0.026 (0.129)
```

#### space_config c1 file: config_4_3.json

Expected output at the 200 step, estimated time to execute 1K subnets is (0.144):

```bash
Stage: [3] Epoch: [0][200/1000] Time(1639706693.9953368): 0.349 (0.423) Epoch time [hr]: 0.029 (0.144)
```

#### space_config c2 file: config_4_2.json

Expected output at the 200 step, estimated time to execute 1K subnets is (0.153):

```bash
Stage: [3] Epoch: [0][200/1000] Time(1639706892.0565453): 0.323 (0.442) Epoch time [hr]: 0.031 (0.153)
```

#### space_config 4 file: config_4.json

Expected output at the 200 step, estimated time to execute 1K subnets is (0.214):

```bash
Stage: [3] Epoch: [0][200/1000] Time(1639707112.2461872): 1.513 (0.672) Epoch time [hr]: 0.043 (0.214)
```
