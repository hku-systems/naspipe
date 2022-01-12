# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import subprocess
import os
from argparse import ArgumentParser, REMAINDER


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="PyTorch distributed training launch "
                                        "helper utilty that will spawn up "
                                        "multiple distributed processes")

    parser.add_argument("--collectsteps", type=int, default=200,
                        help="The number of steps to collect")    
    # Optional arguments for the launch helper
    parser.add_argument("--nnodes", type=int, default=1,
                        help="The number of nodes to use for distributed "
                             "training")
    parser.add_argument("--node_rank", type=int, default=0,
                        help="The rank of the node for multi-node distributed "
                             "training")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="The number of processes to launch on each node, "
                             "for GPU training, this is recommended to be set "
                             "to the number of GPUs in your system so that "
                             "each process can be bound to a single GPU.")

    # positional
    parser.add_argument("training_script", type=str,
                        help="The full path to the single GPU training "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the "
                             "training script")

    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()



def execute(cmd, processes):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    processes.append(popen)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    # return_code = popen.wait()
    # if return_code:
    #     raise subprocess.CalledProcessError(return_code, cmd)

def main():
    print("Experiment Start.")
    args = parse_args()

    # world size in terms of number of processes
    dist_world_size = args.nproc_per_node * args.nnodes

    # set PyTorch distributed related environmental variables
    processes = []

    for local_rank in range(0, args.nproc_per_node):
        # each process's rank
        dist_rank = args.nproc_per_node * args.node_rank + local_rank

        # spawn the processes
        cmd = [sys.executable,
               "-u",
               args.training_script,
               "--rank={}".format(dist_rank),
               "--local_rank={}".format(local_rank)] + args.training_script_args

        if dist_rank == dist_world_size-1: 
            for output in execute(cmd, processes):
                #print(output)
                if "Stage: [3] Epoch: [0]["+str(args.collectsteps) in output:
                    print("Experiment Stop.")
                    print("Steps: "+str(args.collectsteps)+ " Time: "+output[-15:-10])
                    for p in processes:
                        p.kill()

        else:
            process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL)
            processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            if process.returncode == -9:
                print("Process cleaned by SIGKILL.")
            else:
                # print("Error Code:"process.returncode)
                print("Errors encounterred. Please manually kill all processes by [sudo pkill -9 python] and then restart the experiment.")
                raise subprocess.CalledProcessError(returncode=process.returncode,
                                            cmd=cmd)



if __name__ == "__main__":
    main()

