# Artifact Evaluation

[![DOI](https://zenodo.org/badge/433303490.svg)](https://zenodo.org/badge/latestdoi/433303490)


Artifact Ver 0.2 - Functional Version 

## Dataset and AE hosts

You can download the dataset via this link: https://drive.google.com/file/d/1QztbVol4kaQjIL3lpuGvLkBRTB_6Fd1-/view?usp=sharing

You can unzip it to naspipe/[experiment name]/translation/data.

The final data direction should looks like: naspipe/[experiment name]/translation/data/wmt14_en_de_joined_dict/...

If reviewers need a bare metal host to evaluate our artifact, please leave a message in the hotcrp.

## Quick Hints

If you want to exit an experiment quickly, please start another shell and use:

```
sudo pkill -9 python
```

## Installation 


Pull and run PyTorch official image:
```bash
docker pull pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel
```
```bash
cd naspipe/
nvidia-docker run -it -v $PWD:/workspace --net=host --ipc=host pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel
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

Please make sure the data is copied to naspipe/[experiment name]/translation/data/wmt14_en_de_joined_dict/...

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

Please make sure the data is copied to naspipe/[experiment name]/translation/data/wmt14_en_de_joined_dict/...

### Experiment 1:

#### Run sequential execution (no parallel, equivalant to single GPU training):

```bash
python -m launch --nnodes 1 --node_rank 0 --nproc_per_node 4 main_with_runtime_single.py --data_dir data/wmt14_en_de_joined_dict --master_addr localhost --module gpus=4 --checkpoint_dir output --distributed_backend gloo -b 3840 --lr 0.000060 --lr_policy polynomial --weight-decay 0.000000 --epochs 10 --print-freq 10 --verbose 0 --num_ranks_in_server 4 --config_path gpus=4/mp_conf.json
```

Outputs (forward losses) between step 96-100:

```bash
Stage: [3] Epoch: [0][96/49372] Time: 1.828 (1.748)     Id: 95  Tokens: 2976    Output: 32363.60546875000000000000000000000000
Stage: [3] Epoch: [0][97/49372] Time: 1.689 (1.747)     Id: 96  Tokens: 2880    Output: 31520.35351562500000000000000000000000
Stage: [3] Epoch: [0][98/49372] Time: 1.893 (1.749)     Id: 97  Tokens: 3552    Output: 39054.67968750000000000000000000000000
Stage: [3] Epoch: [0][99/49372] Time: 1.921 (1.751)     Id: 98  Tokens: 3456    Output: 37461.26562500000000000000000000000000
Stage: [3] Epoch: [0][100/49372]        Time: 1.926 (1.752)     Id: 99  Tokens: 3520    Output: 39656.17968750000000000000000000000000
```
Outputs (forward losses) between step 196-200:
```bash
Stage: [3] Epoch: [0][196/49372]        Time: 1.633 (1.752)     Id: 195 Tokens: 2208    Output: 26274.00390625000000000000000000000000
Stage: [3] Epoch: [0][197/49372]        Time: 1.901 (1.753)     Id: 196 Tokens: 3200    Output: 30433.37109375000000000000000000000000
Stage: [3] Epoch: [0][198/49372]        Time: 1.972 (1.754)     Id: 197 Tokens: 3328    Output: 40601.20703125000000000000000000000000
Stage: [3] Epoch: [0][199/49372]        Time: 1.838 (1.755)     Id: 198 Tokens: 2912    Output: 33449.57421875000000000000000000000000
Stage: [3] Epoch: [0][200/49372]        Time: 1.781 (1.755)     Id: 199 Tokens: 2912    Output: 32267.23437500000000000000000000000000
```

#### Run parallel execution (under search space c0):

```bash
python -m launch --nnodes 1 --node_rank 0 --nproc_per_node 4 main_with_runtime.py --data_dir data/wmt14_en_de_joined_dict --master_addr localhost --module gpus=4 --checkpoint_dir output --distributed_backend gloo -b 3840 --lr 0.000060 --lr_policy polynomial --weight-decay 0.000000 --epochs 10 --print-freq 10 --verbose 0 --num_ranks_in_server 4 --config_path gpus=4/mp_conf.json
```

Outputs (forward losses) between step 96-100:

(Note that: the parallel execution might be re-ordered, please match Output with Id.)

```bash
Stage: [3] Epoch: [0][96/49372] Time: 1.179 (0.924)     Id: 96  Tokens: 2880    Output: 31520.35351562500000000000000000000000
Stage: [3] Epoch: [0][97/49372] Time: 0.842 (0.924)     Id: 95  Tokens: 2976    Output: 32363.60546875000000000000000000000000
Stage: [3] Epoch: [0][98/49372] Time: 1.023 (0.925)     Id: 97  Tokens: 3552    Output: 39054.67968750000000000000000000000000
Stage: [3] Epoch: [0][99/49372] Time: 0.699 (0.922)     Id: 101 Tokens: 2400    Output: 29934.16992187500000000000000000000000
Stage: [3] Epoch: [0][100/49372]        Time: 1.329 (0.926)     Id: 98  Tokens: 3456    Output: 37461.26562500000000000000000000000000
```

Outputs (forward losses) between step 196-200:

(Note that: the parallel execution might be re-ordered, please match Output with Id.)

```bash
Stage: [3] Epoch: [0][196/49372]        Time: 0.790 (0.966)     Id: 200 Tokens: 2688    Output: 30363.91601562500000000000000000000000
Stage: [3] Epoch: [0][197/49372]        Time: 1.119 (0.967)     Id: 193 Tokens: 3840    Output: 42197.10546875000000000000000000000000
Stage: [3] Epoch: [0][198/49372]        Time: 1.385 (0.969)     Id: 197 Tokens: 3328    Output: 40601.20703125000000000000000000000000
Stage: [3] Epoch: [0][199/49372]        Time: 0.941 (0.969)     Id: 196 Tokens: 3200    Output: 30433.37109375000000000000000000000000
Stage: [3] Epoch: [0][200/49372]        Time: 1.018 (0.969)     Id: 201 Tokens: 3456    Output: 43732.59765625000000000000000000000000
```

### Experiment 2:

In Experiment 2, we provide the original environment where we conducted our performance evaluation. The throughput trend matches our results in Figure 5 (although Figure 5 was conducted on 8 GPUs).

Run 200 steps is enought to get a stable throughput.

Pull the transformer image with apex installed:

```bash
docker pull zsxhku/transformer:apex
```

Start docker under the AE directory:
```bash
cd naspipe/
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

### Experiment 2 Auto-generate Figure 5

[NOTE!] Please make sure the data is copied to naspipe/baselines/translation/data/wmt14_en_de_joined_dict/...

```bash
cd naspipe/
nvidia-docker run -it -v $PWD:/workspace --net=host --ipc=host zsxhku/transformer:apex
cd /workspace/
```

#### Run experiments
Run Figure 5 script and save the log to a file:

```bash
./figure5.sh | tee output
```

If you encounter any errors, clean the processes by running the naspipe/clean.sh on host (not inside the docker); then re-run the script. You can manually select experiments (by deleting unwanted experiments from the script).

#### Generate Figures

Make sure the matplotlib is installed:

```bash
python -m pip install matplotlib
```

Generate Figure5:

```bash
python gen_figure.py output figure5
```

Then you will get figure5.pdf.

#### Get Figure PDF from the server via Email
```bash
sudo apt-get install mailutils
```
For the options, choose Internet Site.

Use scp to copy the generated figures;

Or you can use the below command to send the generated PDFs to your email (remember to set your email address).

```bash
echo "Figure 5" | mail -s "Experiment Figure 5" your_email -A figure5.pdf
```
