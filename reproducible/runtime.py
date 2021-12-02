# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import collections
import itertools
import time
from datetime import timedelta
import torch
import torch.distributed as dist
import json

import communication
import runtime_utilities


import threading
import queue

#import apex_C

IMAGE_CLASSIFICATION = "image_classification"
TRANSLATION = "translation"
SPEECH_TO_TEXT = "speech_to_text"


SWAP_IN = "_swap_in"
SWAP_OUT = "_swap_out"
EXIT_QUEUE = "_exit_queue"


SEED = 123 # or whatever you choose
# random.seed(SEED) # if you're using random
# np.random.seed(123) # if you're using numpy
torch.manual_seed(SEED) # torch.cuda.manual_seed_all(SEED) is not required
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def context_manager(queue, modules, stage_id):
    
    print("CxtM: Starting message queue...")

    cuda_stream_in_parameter = torch.cuda.Stream()
    cuda_stream_out_parameter = torch.cuda.Stream()

    device_cuda = torch.device("cuda:0")
    device_cpu = torch.device("cpu")

    stage_id = stage_id

    print("Initialize lock")
    while True:
        msg = queue.get()
        operation = msg[0]
        payload = msg[1]

        if operation == SWAP_IN:
            #print("CtxM: Swapping in...")
            seq_id = int(payload)
            stage = modules[0] ## get the stage 

            start = stage.start
            if stage.start == 0: ## first stage two embeddings 
                #assert stage.end > 1
                #input0 = stage.module[0](input0)
                #input1 = stage.module[1](input1)
                start = 2
            print("CtxM: Swapping IN on stage "+str(stage_id)+" with payload "+str(stage.idx[start:stage.end]))
            for i in range(start, stage.end):
                m = modules[0].module[i]
                if i > 1 and i < len(stage.module) - 1:
                    temp_idx = stage.idx[i - 2]
                    
                    layer = m.layer[temp_idx]
                    
                    #layer.cuda()
                    for p in layer.parameters():
                        #if not p.data.is_cuda:                       
                        #    p.data = p.data.pin_memory()
                        
                        with torch.cuda.stream(cuda_stream_in_parameter):
                            p.data = p.data.cuda(non_blocking=True)

                    layer = m.layer_norm

                    for p in layer.parameters():
                        
                        #if not p.data.is_cuda:                       
                        #    p.data = p.data.pin_memory()
                        with torch.cuda.stream(cuda_stream_in_parameter):
                            p.data = p.data.cuda(non_blocking=True)
                        #temp.share_memory_()
                        # p.data = temp

                    #return m(input1)

        elif operation == SWAP_OUT:

            idx = json.loads(payload)
            stage = modules[0] ## get the stage 

            start = stage.start
            if stage.start == 0: ## first stage two embeddings 
                #assert stage.end > 1
                #input0 = stage.module[0](input0)
                #input1 = stage.module[1](input1)
                start = 2
            
            print("CtxM: Swapping OUT on stage "+str(stage_id)+" with payload "+str(idx[start:stage.end]))

            for i in range(start, stage.end):
                m = modules[0].module[i]
                if i > 1 and i < len(stage.module) - 1: # check again 
                    temp_idx = int(idx[i - 2])
                    
                    #if type(layer).__name__ != "Linear": 
                    layer = m.layer[temp_idx]
                    for p in layer.parameters():
                        with torch.cuda.stream(cuda_stream_in_parameter):
                            p.data = p.data.to(device_cpu, non_blocking=True).pin_memory()

            with torch.cuda.stream(cuda_stream_in_parameter):
                e = torch.cuda.Event()
                e.record()
                e.synchronize()


        elif operation == EXIT_QUEUE:
            print("CxtM: Exiting Context Manager...")
            break
        else:
            print("CxtM: Invalid Operation for Context Manager Process!")

class ModulesWithDependencies:
    def __init__(self, modules_with_dependencies):
        self._modules = []
        self._all_input_names = []
        self._all_output_names = []
        for (module, input_names, output_names) in modules_with_dependencies:
            self._modules.append(module)
            self._all_input_names.append(input_names)
            self._all_output_names.append(output_names)

    def modules(self):
        return self._modules

    def all_input_names(self):
        return self._all_input_names

    def all_output_names(self):
        return self._all_output_names

    def is_input_tensor(self, tensor_name):
        for module_input_names in self._all_input_names:
            if tensor_name in module_input_names:
                return True
        return False

cnt = 0
class StageRuntime:
    def __init__(self, model, distributed_backend, fp16, loss_scale,
                 training_tensor_shapes, eval_tensor_shapes,
                 training_tensor_dtypes, inputs_module_destinations,
                 target_tensor_names, configuration_maps, master_addr,
                 rank, local_rank, num_ranks_in_server, verbose_freq,
                 model_type, enable_recompute=False):
        # Metadata needed for forward and backward pass within this stage.
        self.tensors = []
        self.gradients = {}
        self.distributed_backend = distributed_backend
        self.fp16 = fp16
        self.loss_scale = loss_scale
        self.training_tensor_shapes = training_tensor_shapes
        self.eval_tensor_shapes = eval_tensor_shapes
        self.training_tensor_dtypes = training_tensor_dtypes
        self.model_type = model_type
        self.target_tensor_names = target_tensor_names
        self.grad = None
        self.master_addr = master_addr

        self.initialize(model, inputs_module_destinations, configuration_maps,
                        master_addr, rank, local_rank, num_ranks_in_server)

        self.verbose_freq = verbose_freq
        self.forward_only = False

        self.forward_stats = runtime_utilities.RuntimeStats(forward=True)
        self.backward_stats = runtime_utilities.RuntimeStats(forward=False)

        # Enable recomputation to prevent the need to save activations
        # computed from the forward pass for the backward pass.
        self.enable_recompute = enable_recompute

        # Disable recomputation for the last stage.
        if rank == num_ranks_in_server - 1:
            self.enable_recompute = False

        subnets = json.load(open('config_4_4.json'))
        self.ops = []
        self.partitions = []
        for sub in subnets:
            sub = json.loads(sub)
            self.ops.append(sub['op'])
            self.partitions.append(sub['part'])

        self.buffer_list = []
        self.wait_list = []
        self.mem_list = []
        self.input_index = 0
        self.current_fwd_idx = 0
        self.current_bwd_idx = 0 
        self.wait_upper = 0

    def initialize(self, model, inputs_module_destinations,
                   configuration_maps, master_addr, rank,
                   local_rank, num_ranks_in_server):
        self.send_ranks = {}
        self.receive_ranks = {}
        self.rank = rank
        self.local_rank = local_rank
        self.stage = None
        self.tensor_tags = {}
        self.forward_minibatch_id = 0
        self.backward_minibatch_id = 0
        self.criterion_input_name = str(model[-1][1][0])

        tensor_tag = 1
        for (_, input_tensors, output_tensors) in model:
            for input_tensor in input_tensors:
                if input_tensor not in self.tensor_tags:
                    self.tensor_tags[input_tensor] = tensor_tag
                    tensor_tag += 1
            for output_tensor in output_tensors:
                if output_tensor not in self.tensor_tags:
                    self.tensor_tags[output_tensor] = tensor_tag
                    tensor_tag += 1
        for target_tensor_name in sorted(self.target_tensor_names):
            self.tensor_tags[target_tensor_name] = tensor_tag
            tensor_tag += 1
        self.tensor_tags["ack"] = tensor_tag
        tensor_tag += 1

        self.tensor_tags["control"] = tensor_tag 
        tensor_tag += 1

        module_to_stage_map = configuration_maps['module_to_stage_map']
        stage_to_rank_map = configuration_maps['stage_to_rank_map']
        stage_to_depth_map = configuration_maps['stage_to_depth_map']

        if module_to_stage_map is None:
            # If IP addresses not specified, resort to all layers on
            # single machine.
            assert self.rank is None
            self.modules_with_dependencies = ModulesWithDependencies(model)
            self.is_criterion = True
            self.rank_in_stage = 0
            self.num_ranks = 1
            self.num_ranks_in_first_stage = 1
            self.num_ranks_in_previous_stage = 0
            self.num_ranks_in_next_stage = 0
            self.num_stages = 1
            self.num_ranks_in_stage = 1
            self.num_warmup_minibatches = 0
            self.comm_handler = None
        else:
            assert len(module_to_stage_map) == len(model)
            assert self.rank is not None

            stage_to_module_map = collections.defaultdict(list)
            for module in range(len(module_to_stage_map)):
                stage_to_module_map[module_to_stage_map[module]].append(module)

            rank_to_stage_map = {}
            for stage in stage_to_rank_map:
                for rank in stage_to_rank_map[stage]:
                    rank_to_stage_map[rank] = stage

            # Now, use this mapping to determine the modules contained in
            # each stage.
            assert 0 <= self.rank < len(rank_to_stage_map)
            self.num_ranks = len(rank_to_stage_map)
            self.num_stages = len(stage_to_module_map)
            self.stage = rank_to_stage_map[self.rank]
            self.rank_in_stage = stage_to_rank_map[self.stage].index(self.rank)
            self.num_ranks_in_stage = len(stage_to_rank_map[self.stage])
            self.num_ranks_in_first_stage = len(stage_to_rank_map[0])
            self.num_ranks_in_previous_stage = 0
            self.ranks_in_previous_stage = []
            if self.stage > 0:
                self.num_ranks_in_previous_stage = len(
                    stage_to_rank_map[self.stage - 1])
                self.ranks_in_previous_stage = stage_to_rank_map[self.stage - 1]
            self.num_ranks_in_next_stage = 0
            self.ranks_in_next_stage = []
            if self.stage < self.num_stages - 1:
                self.num_ranks_in_next_stage = len(
                    stage_to_rank_map[self.stage + 1])
                self.ranks_in_next_stage = stage_to_rank_map[self.stage + 1]
            modules = stage_to_module_map[self.stage]
            self.modules_with_dependencies = ModulesWithDependencies(
                [model[module] for module in modules])
            self.is_criterion = self.stage == (self.num_stages - 1)
            if stage_to_depth_map is not None:
                self.num_warmup_minibatches = stage_to_depth_map[
                    str(self.stage)]
            else:
                self.num_warmup_minibatches = self.num_ranks - 1
                for i in range(self.stage):
                    self.num_warmup_minibatches -= len(
                        stage_to_rank_map[i])
                self.num_warmup_minibatches = self.num_warmup_minibatches // \
                    self.num_ranks_in_stage

            # To determine where tensors should be sent and received, first
            # determine the "producing" and "consuming" module IDs of each
            # tensor. We then use the corresponding machine ranks to send
            # and receive tensors.
            master_port = 12345
            self.comm_handler = communication.CommunicationHandler(
                master_addr=master_addr,
                master_port=master_port,
                rank=self.rank,
                local_rank=self.local_rank,
                num_ranks_in_server=num_ranks_in_server,
                world_size=self.num_ranks,
                fp16=self.fp16,
                backend=self.distributed_backend)

            for i in range(len(model)-1):
                for tensor_name in model[i][2]:
                    if tensor_name in model[i+1][1]:
                        if module_to_stage_map[i] == \
                            module_to_stage_map[i+1]:
                            continue
                        # For now, assume that each stage is served by only
                        # a single machine.
                        if module_to_stage_map[i+1] == self.stage:
                            self.receive_ranks[tensor_name] = \
                                stage_to_rank_map[module_to_stage_map[i]]
                        if module_to_stage_map[i] == self.stage:
                            self.send_ranks[tensor_name] = \
                                stage_to_rank_map[module_to_stage_map[i+1]]

            for model_inputs in inputs_module_destinations.keys():
                destination_stage = module_to_stage_map[
                    inputs_module_destinations[model_inputs]]
                if destination_stage > self.stage:
                    self.send_ranks[model_inputs] = \
                        self.ranks_in_next_stage

                if 0 < self.stage <= destination_stage:
                    self.receive_ranks[model_inputs] = \
                        self.ranks_in_previous_stage

                if destination_stage > 0:
                    if model_inputs not in self.tensor_tags:
                        self.tensor_tags[model_inputs] = tensor_tag
                        tensor_tag += 1

        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i] = modules[i].cuda()
            # if self.fp16:
            #     import apex.fp16_utils as fp16_utils
            #     modules[i] = fp16_utils.BN_convert_float(modules[i].half())
        # print(next(modules[0].module[5].layer_norm.parameters()).device)

        # Initialize all groups in the same order on every worker.
        if stage_to_rank_map is not None:
            groups = []
            for stage in range(self.num_stages):
                ranks = stage_to_rank_map[stage]
                if len(ranks) > 1:
                    groups.append(dist.new_group(ranks=ranks))
                else:
                    groups.append(None)
            group = groups[self.stage]
        else:
            group = None

        # self.modules_with_dependencies contains a list of PyTorch
        # modules, along with a list of user-defined input and output
        # tensor names. We use our module_executor.ModuleExecutor
        # class to wrap these dependencies, and use run_forward and
        # run_backward methods downstream.
        num_parameters = 0
        for i in range(len(modules)):
            if group is not None:
                if ((i < (len(modules)-1) and self.is_criterion)
                    or not self.is_criterion):
                    num_parameters += \
                        sum(x.size()[0] * x.size()[1]
                            if len(x.size()) > 1 else x.size()[0]
                            for x in modules[i].parameters() if x.size())
                    modules[i] = torch.nn.parallel.DistributedDataParallel(
                        modules[i],
                        process_group=group,
                        device_ids=[local_rank],
                        output_device=local_rank)
        if self.num_ranks_in_stage > 1:
            module_size = 4. * num_parameters
            print("Replicating stage: ranks=%d, module_size=%.3f" % (
                self.num_ranks_in_stage, module_size))

        if self.fp16:
            self.master_parameters = []
            self.model_parameters = []
            for i in range(len(modules)):
                import apex.fp16_utils as fp16_utils
                module_parameters, module_master_parameters = \
                    fp16_utils.prep_param_lists(modules[i])
                self.master_parameters.extend(module_master_parameters)
                self.model_parameters.extend(module_parameters)
        else:
            self.master_parameters = list(self.parameters())
            self.model_parameters = None

        if self.stage > 0:
            self.receive_ranks["control"]=stage_to_rank_map[self.stage-1]

        if self.stage < self.num_stages-1:
            self.send_ranks["control"]=stage_to_rank_map[self.stage+1]

        if self.comm_handler is not None:
            self.comm_handler.initialize(
                self.receive_ranks,
                self.send_ranks,
                self.tensor_tags,
                self.target_tensor_names,
                self.training_tensor_dtypes,
                self.rank_in_stage,
                self.num_ranks_in_stage,
                self.ranks_in_previous_stage,
                self.ranks_in_next_stage)


    def set_share(self):
        #print("Start share_memory_() settings")

        #mp.set_start_method('spawn', force=True)

        #self.modules()[0].param_share()
        
        #print("Finished share_memory_() settings.")

        print("Initialize event queue.")
        self.cm_queue = queue.Queue()

        self.cm_process = threading.Thread(target=context_manager, args=(self.cm_queue, self.modules_with_dependencies.modules(), self.stage, ))
        self.cm_process.start()

        ## init kv store for inter-stage simple message pass communication
        if self.stage == 0:
            self.kv_store = dist.TCPStore("127.0.0.1", 1234, self.num_ranks, True, timedelta(seconds=30))
            self.kv_store.set("test_key", "message passing is set!")
        else:
            self.kv_store = dist.TCPStore(self.master_addr, 1234, self.num_ranks, False)
            print(self.kv_store.get("test_key"))


    @property
    def target(self):
        return self.tensors[-1]["target"]

    def modules(self):
        return self.modules_with_dependencies.modules()

    def parameters(self):
        parameter_iterators = []
        for module in self.modules_with_dependencies.modules():
            #print(list(module.modules())[0])
            parameter_iterators.append(module.parameters())
        return itertools.chain(*parameter_iterators)

    def state_dict(self):
        state_dict = collections.OrderedDict()
        for i, module in enumerate(self.modules_with_dependencies.modules()):
            state_dict["module%d" % i] = module.state_dict()
        if self.fp16:
            state_dict["master_parameters"] = self.master_parameters
        return state_dict

    def load_state_dict(self, state_dict):
        for i, module in enumerate(self.modules_with_dependencies.modules()):
            module.load_state_dict(state_dict["module%d" % i])
        if self.fp16:
            saved_master_parameters = state_dict["master_parameters"]
            for master_parameter, saved_master_parameter in zip(
                self.master_parameters, saved_master_parameters):
                master_parameter.data.copy_(saved_master_parameter.data)

    def cuda(self):
        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i] = modules[i].cuda()

    def zero_grad(self):
        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].zero_grad()

    def train(self, num_iterations):
        self.tensors = []
        self.gradients = {}
        self.control = []
        self.tensor_shapes = self.training_tensor_shapes
        self.forward_only = False

        self.forward_minibatch_id = 0
        self.backward_minibatch_id = 0

        if self.comm_handler is not None:
            self.comm_handler.set_tensor_shapes(self.tensor_shapes)
            self.comm_handler.start_helper_threads(
                num_iterations, forward_only=False)

        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].train()

    def eval(self, num_iterations):
        self.tensors = []
        self.gradients = {}
        self.tensor_shapes = self.eval_tensor_shapes
        self.tensor_shapes["ack"] = (1,)
        self.forward_only = True

        self.forward_minibatch_id = 0
        self.backward_minibatch_id = 0

        if self.comm_handler is not None:
            self.comm_handler.set_tensor_shapes(self.tensor_shapes)
            self.comm_handler.start_helper_threads(
                num_iterations, forward_only=True)

        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].eval()

    def set_loader(self, loader):
        if loader is not None:
            self.loader_iter = loader.next_epoch_itr()
        else:
            self.loader_iter = None

    def set_loss_scale(self, loss_scale):
        self.loss_scale = loss_scale

    def receive_tensors_forward(self): #, blocking = True):


        ### If current is empty:
        ## if self.tensors[-1]["input0"] = None:

        # if not blocking and self.comm_handler.get_recv_queue = None:
        #     continue 



        # If not the first stage, and nothing to receive, skip this function
        # If tensors NONE, should wait
        if len(self.tensors) != 0 and self.loader_iter is None and self.comm_handler.get_recv_queue().empty():
            return False

        # if self.loader_iter is None:
        #     print("Stage ", self.stage, " : Receive queue size", self.comm_handler.get_recv_queue().size())

        if self.forward_only and len(self.tensors) > 0:
            self.tensors.pop(0)
        self.tensors.append({})
        self.control.append({})

        self.control[-1]["forward_receive"]=None

        if self.loader_iter is not None:
            input = next(self.loader_iter)
            if self.model_type == TRANSLATION:
                #print(input.keys())
                inputs = input['net_input']
                target = input['target']
                self.tensors[-1]["input0"] = inputs['src_tokens'].cuda(non_blocking=True)
                self.tensors[-1]["input1"] = inputs['prev_output_tokens'].cuda(non_blocking=True)
                self.tensors[-1]["target"] = target.cuda()
                self.tensors[-1]["ntokens"] = torch.zeros(input['ntokens']).cuda()

            elif self.model_type == IMAGE_CLASSIFICATION:
                (input, target) = input
                if self.fp16:
                    input = input.half()
                self.tensors[-1]["input0"] = input.cuda(non_blocking=True)
                self.tensors[-1]["target"] = target.cuda(non_blocking=True)
            elif self.model_type == SPEECH_TO_TEXT:
                input, target, input_percentages, target_sizes = input
                input_sizes = input_percentages.mul_(int(input.size(3))).int()
                self.tensors[-1]["input0"] = input.cuda(non_blocking=True)
                self.tensors[-1]["input1"] = input_sizes.cuda(non_blocking=True)
                self.tensors[-1]["target"] = target.cuda(non_blocking=True)
                self.tensors[-1]["target_length"] = target_sizes.cuda(
                    non_blocking=True)
            self.tensors[-1]["seq"] = self.input_index
            self.input_index += 1
        else:
            # Receive all required tensors from upstream machines.
            for input_name in self.receive_ranks:
                if input_name == "ack":
                    continue
                if input_name == "control":
                    #print("Received control message")
                    self.control[-1]["forward_receive"] = \
                        self.comm_handler.recv(
                            input_name,
                            forward_minibatch_id=self.forward_minibatch_id,
                            backward_minibatch_id=self.backward_minibatch_id,
                            backward=False)
                    continue

                self.tensors[-1][input_name] = \
                    self.comm_handler.recv(
                        input_name,
                        forward_minibatch_id=self.forward_minibatch_id,
                        backward_minibatch_id=self.backward_minibatch_id,
                        backward=False)

                self.forward_stats.stats['receive_tensors_size'] += \
                    (self.tensors[-1][input_name].element_size() *
                    self.tensors[-1][input_name].nelement())

            self.tensors[-1]["seq"] = self.control[-1]["forward_receive"].tolist()[-1][0]

            
            # Used to track where to receive forward from.
            self.comm_handler.increment_messaging_index(
                sending=False)

        # print("Stage ", self.stage, ": receive forward ", self.tensors[-1]["seq"])

        return True

    def send_tensors_forward(self):
        # Send all required tensors downstream.
        for output_name in self.send_ranks:
            if output_name == "ack":
                continue

            if output_name == "control":
                # print("Stage ", self.stage, ": send forward ", self.control[-1]["forward_send"].tolist()[-1][0])
                self.comm_handler.send(
                    output_name,
                    self.control[-1]["forward_send"],
                    forward_minibatch_id=self.forward_minibatch_id,
                    backward_minibatch_id=self.backward_minibatch_id,
                    backward=False)  
                continue

            tensors = None
            for idx, t in enumerate(self.tensors):
                if t["seq"] == self.current_fwd_idx:
                    #print("send fwd tensors seq", t["seq"])
                    tensors = self.tensors[idx]
                    break


            self.comm_handler.send(
                output_name,
                tensors[output_name],
                forward_minibatch_id=self.forward_minibatch_id,
                backward_minibatch_id=self.backward_minibatch_id,
                backward=False)

            self.forward_stats.stats['send_tensors_size'] += \
                (tensors[output_name].element_size() *
                 tensors[output_name].nelement())

    def receive_tensors_backward(self):
        # Receive all required gradients from downstream
        # machines.

        # if self.stage == 1:
        #     q = self.comm_handler.get_recv_queue(forward_minibatch_id=self.forward_minibatch_id, backward_minibatch_id=self.backward_minibatch_id, backward=True)
        #     print(q, q.empty())
        #     print(q.queue[0][0])

        if not self.is_criterion and self.comm_handler.get_recv_queue(forward_minibatch_id=self.forward_minibatch_id, backward_minibatch_id=self.backward_minibatch_id, backward=True).empty():
            return False

        self.control[-1]["backward_receive"]=None

        for output_name in self.send_ranks:
            if output_name in self.target_tensor_names:
                continue

            if output_name == "control":
                    #print("Stage ", self.stage, " Received backward control message")
                    self.control[-1]["backward_receive"] = \
                        self.comm_handler.recv(
                            output_name,
                            forward_minibatch_id=self.forward_minibatch_id,
                            backward_minibatch_id=self.backward_minibatch_id,
                            backward=True)

                    self.gradients["seq"] = self.control[-1]["backward_receive"].tolist()[-1][0]

                    self.current_bwd_idx = self.gradients["seq"]

                    # print("Stage ", self.stage, ": receive backward ", self.current_bwd_idx, flush=True)

                    continue

            self.gradients[output_name] = \
                self.comm_handler.recv(
                    output_name,
                    forward_minibatch_id=self.forward_minibatch_id,
                    backward_minibatch_id=self.backward_minibatch_id,
                    backward=True)

            self.backward_stats.stats['receive_tensors_size'] += \
                 (self.gradients[output_name].element_size() *
                  self.gradients[output_name].nelement())

        #self.control.pop(0)

        return True

    def send_tensors_backward(self):
        # Send all required gradients upstream.
        for input_name in self.receive_ranks:
            if input_name in self.target_tensor_names:
                continue

            if input_name == "control":
                # print("Stage ", self.stage, ": send backward ", self.control[-1]["backward_send"].tolist()[-1][0], flush=True)
                self.comm_handler.send(
                    input_name,
                    self.control[-1]["backward_send"],
                    forward_minibatch_id=self.forward_minibatch_id,
                    backward_minibatch_id=self.backward_minibatch_id,
                    backward=True)  
                continue

            self.comm_handler.send(
                input_name,
                self.gradients[input_name],
                forward_minibatch_id=self.forward_minibatch_id,
                backward_minibatch_id=self.backward_minibatch_id,
                backward=True)

            self.backward_stats.stats['send_tensors_size'] += \
                (self.gradients[input_name].element_size() *
                 self.gradients[input_name].nelement())

        if self.num_ranks_in_previous_stage > 0:
            # Used to track where to send tensors in the
            # backward pass.
            self.comm_handler.increment_messaging_index(
                sending=True)

    def run_forward(self, idx=None, recompute_step=False):
        """Run forward pass.
        """
        # Receive tensors from previous worker.
        #self.receive_tensors_forward()

        #print("Stage ", self.stage, "queue size", self.comm_handler.get_recv_queue().size())
        # Run forward pass.
        if idx is None:
            seq_id = self.forward_minibatch_id
        else:
            self.current_fwd_idx = idx
            seq_id = idx

        for idx, t in enumerate(self.tensors):
            if t["seq"] == seq_id:
                #print("fwd tensors seq", t["seq"])
                tensors = self.tensors[idx]
                break

        self.n_tokens = tensors["ntokens"].size(0)


        #print("Forward: Stage ", self.stage, "keys ", tensors.keys(), " seq ", tensors["seq"])
        #tensors = self.tensors[-1]
        #Receive forward stats from the previous worker 
        #print("forward receive")
        #print(self.control[-1]["forward_receive"])


        ### Set Runtime Indexes
        modules = self.modules_with_dependencies.modules()
        # modules[0].idx = self.ops[seq_id]
        op = self.ops[seq_id]
        eop = op[:24]
        dop = op[24:]
        # for i in range(len(eop)):
        #     while eop[i] >= 23:
        #         eop[i] -= 23
        # for i in range(len(dop)):
        #     while dop[i] >= 24:
        #         dop[i] -= 23
        #print(modules[0 ])
        modules[0].idx = eop + dop
        modules[0].seq_id = seq_id

        start_time = time.time()
        self._run_forward(tensors)
        self.fwd_time = time.time()-start_time

        # Set control message
        fwd_list = None
        if self.control[-1]["forward_receive"] is not None:
            fwd_list = self.control[-1]["forward_receive"].tolist()[-1]
            flag = 2
            while True:
                if fwd_list[flag]==0:
                    break
                flag += 1
            fwd_list[flag] = int(self.fwd_time * 1000000)

            fwd_list[0] = seq_id
        else:
            fwd_list = [int(0)]*100
            fwd_list[2] = int(self.fwd_time * 1000000)

            fwd_list[0] = seq_id

        self.control[-1]["forward_send"]=torch.Tensor([fwd_list]).type(torch.int).cuda()
        #print("forward send")
        #print(self.control[-1]["forward_send"])

        # Send tensors forward.
        self.send_tensors_forward()
        if self.verbose_freq > 0 and self.forward_minibatch_id % self.verbose_freq == 0:
            self.forward_stats.print_stats()
        self.forward_stats.reset_stats()
        self.forward_minibatch_id += 1

    def _run_forward(self, tensors):
        # Perform forward pass through model (self.modules_with_dependencies already
        # has modules in topological order).
        modules = self.modules_with_dependencies.modules()
        #modules[0].idx = self.ops[self.forward_minibatch_id]
        #self.ops.pop(0)
        part = self.partitions[self.forward_minibatch_id]
        #self.partitions.pop(0)
        #modules[0].start = part[self.rank]
        #modules[0].end = part[self.rank + 1]
        all_input_names = self.modules_with_dependencies.all_input_names()
        all_output_names = self.modules_with_dependencies.all_output_names()
        for i, (module, input_names, output_names) in \
                enumerate(zip(modules, all_input_names, all_output_names)):
            if i == (len(modules) - 1) and self.is_criterion:
                # If layer is criterion (loss).
                if self.model_type == SPEECH_TO_TEXT:
                    output = tensors["output"].transpose(0, 1).float()
                    output_sizes = tensors["output_sizes"].cpu()
                    target = tensors["target"].cpu()
                    target_sizes = tensors["target_length"].cpu()
                    input0_size = tensors["input0_size"].cpu()
                    module_outputs = [module(output, target, output_sizes, target_sizes) / input0_size[0]]
                else:
                    module_outputs = [module(tensors[input_name],
                                             tensors["target"])
                                      for input_name in input_names]
                    module_outputs = [sum(module_outputs)]
            else:
                # If layer is non-criterion.
                module_outputs = module(*[tensors[input_name]
                                          for input_name in input_names])
                if not isinstance(module_outputs, tuple):
                    module_outputs = (module_outputs,)
                module_outputs = list(module_outputs)

            for (output_name, module_output) in zip(output_names, module_outputs):
                tensors[output_name] = module_output

        self.output = tensors[input_names[0]]
        if self.is_criterion:
            self.loss = tensors[output_names[0]]
        else:
            self.loss = 1

    def run_backward(self):
        # Receive input gradients needed for backward pass.
        #self.receive_tensors_backward()


        if self.is_criterion:
            self.current_bwd_idx = self.current_fwd_idx
            bwd_seq_id = self.current_fwd_idx
        else:
            #self.current_bwd_idx = self.gradients["seq"]
            bwd_seq_id = self.current_bwd_idx

        #print("backward receive")
        #print(self.control[-1]["backward_receive"])

        # Backward pass through modules in reverse order.
        inputs = {}
        outputs = {}
        input_gradients = {}
        output_gradients = {}

        # Get input and output names spanning all modules in this stage.
        all_input_names_set = set()
        all_output_names_set = set()

        modules = self.modules_with_dependencies.modules()
        all_input_names = self.modules_with_dependencies.all_input_names()
        all_output_names = self.modules_with_dependencies.all_output_names()

        for (input_names, output_names) in zip(all_input_names, all_output_names):
            for input_name in input_names:
                all_input_names_set.add(input_name)
            for output_name in output_names:
                all_output_names_set.add(output_name)



        # tensors = self.tensors.pop(0)

        for idx, t in enumerate(self.tensors):
            #print("bwd tensors seq", t["seq"])
            if t["seq"] == bwd_seq_id:
                tensors = self.tensors.pop(idx)
                break


        self.n_tokens = tensors["ntokens"].size(0)

        #print("Backward: Stage ", self.stage, "keys ", tensors.keys(), " seq ", tensors["seq"])

                
        # Set inputs, outputs, and output_gradients.
        # Only set outputs/output_gradients for tensors that are not inputs of
        # other modules in this stage.
        # Similarly, only set inputs for tensors that are not outputs of other
        # modules in this stage.
        for (module, input_names, output_names) in \
            zip(reversed(modules), reversed(all_input_names), reversed(all_output_names)):
            for output_name in output_names:
                if output_name not in all_input_names_set:
                    if output_name not in self.gradients:
                        output_gradients[output_name] = None
                    else:
                        output_gradients[output_name] = self.gradients[output_name]

                    #print(str(tensors))
                    #print("Stage ", self.stage, " ",output_name)
                    if tensors[output_name].requires_grad:
                        outputs[output_name] = tensors[output_name]
            for input_name in input_names:
                if input_name not in all_output_names_set or input_name in all_output_names[-1]:
                    inputs[input_name] = tensors[input_name]

        # Hook to record input gradients.
        def hook_wrapper(input_name):
            def hook(input_gradient):
                input_gradients[input_name] = input_gradient
            return hook

        for input_name in inputs:
            if input_name != "input0" and input_name != "input1" and input_name != "input2" \
                    and inputs[input_name].requires_grad:
                inputs[input_name].register_hook(hook_wrapper(input_name))

        if "loss" in outputs:
            outputs["loss"] *= self.loss_scale

        #self.zero_grad()
        
        ### Set Runtime Indexes
        modules = self.modules_with_dependencies.modules()
        # modules[0].idx = self.ops[seq_id]
        op = self.ops[bwd_seq_id]
        eop = op[:24]
        dop = op[24:]
        # for i in range(len(eop)):
        #     while eop[i] >= 23:
        #         eop[i] -= 23
        # for i in range(len(dop)):
        #     while dop[i] >= 24:
        #         dop[i] -= 23
        modules[0].idx = eop + dop
        modules[0].seq_id = bwd_seq_id

        # Perform backward pass.
        torch.autograd.backward(tuple([outputs[output_name] for output_name in outputs]),
                                grad_tensors=tuple([output_gradients[output_name]
                                                    for output_name in outputs]))

        # Input tensors don't need gradients.
        for input_name in inputs:
            if not inputs[input_name].requires_grad:
                self.gradients[input_name] = inputs[input_name]
                continue

            if input_name != "input0" and input_name != "input1" and input_name != "input2" and input_name != "input":
                if input_name not in input_gradients:
                    self.gradients[input_name] = inputs[input_name]
                else:
                    self.gradients[input_name] = input_gradients[input_name]


        bwd_list = [int(0)]*100

        bwd_list[0] = bwd_seq_id

        self.control[-1]["backward_send"]=torch.Tensor([bwd_list]).type(torch.int).cuda()
        # self.control[-1]["backward_send"]=torch.zeros([1,100],dtype=torch.int).cuda()
        #print("backward send")
        #print(self.control[-1]["backward_send"])
        
        # Send output gradients.
        self.send_tensors_backward()
        if self.verbose_freq > 0 and self.backward_minibatch_id % self.verbose_freq == 0:
            self.backward_stats.print_stats()
        self.backward_stats.reset_stats()
        self.backward_minibatch_id += 1

    def num_tokens(self):
        return self.n_tokens

    def run_ack(self):
        # No need for ack if running on a single worker.
        if self.rank is None:
            return

        # Receive ack from next stage. Send ack to previous stage.
        if self.stage < (self.num_stages-1):
            self.comm_handler.recv(
                "ack",
                forward_minibatch_id=self.forward_minibatch_id,
                backward_minibatch_id=self.backward_minibatch_id,
                backward=True)
        if self.stage > 0:
            self.comm_handler.send(
                "ack",
                torch.zeros(self.tensor_shapes["ack"],
                            dtype=torch.int64).cuda(),
                forward_minibatch_id=self.forward_minibatch_id,
                backward_minibatch_id=self.backward_minibatch_id,
                backward=True)

            # Used to track where to receive forward from.
            self.comm_handler.increment_messaging_index(sending=True)

        self.backward_minibatch_id += 1

    def wait(self):
        if self.comm_handler is not None:
            self.comm_handler.wait()

    def num_iterations(self, loader_size):
        """ Determines number of iterations for this stage

        TODO: don't currently support uneven configurations.
        """
        if self.stage == 0 or self.stage is None:
            return loader_size

        num_iterations = loader_size * self.num_ranks_in_first_stage
        assert num_iterations % self.num_ranks_in_stage == 0
        num_iterations = num_iterations // self.num_ranks_in_stage

        return num_iterations

    def get_adjusted_learning_rate(self, base_lr):
        if self.stage == 0:
            return base_lr

        adjusted_lr = float(base_lr) * float(self.num_ranks_in_stage) \
                      / float(self.num_ranks_in_first_stage)

        return adjusted_lr
