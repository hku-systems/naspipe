from transformer_oneshot import TransformerModel
from transformer_oneshot import TransformerEncoderLayer
from transformer_oneshot import TransformerDecoderLayer
from fairseq.data import Dictionary
from .transformer import TRANSFORMERPartitioned

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.checkpoint as cp

class Args():
    def __init__(self):
        self.encoder_embed_dim = 1024
        self.decoder_embed_dim = 1024
        self.attention_dropout = 0.1
        self.relu_dropout = 0.1
        self.dropout = 0.3

        self.encoder_layers = 24
        self.decoder_layers = 24
        self.encoder_embed_path = None
        self.encoder_learned_pos = False
        self.decoder_embed_path =None
        self.decoder_learned_pos = False
        self.adaptive_softmax_cutoff = None
        self.share_decoder_input_output_embed = False
        self.no_token_positional_embeddings = False

class Task():
    def __init__(self):
        self.source_dictionary = Dictionary.load('data/wmt14_en_de_joined_dict/dict.en.txt')
        self.target_dictionary = Dictionary.load('data/wmt14_en_de_joined_dict/dict.de.txt')

def arch():
    return "transformer"

class Token_Embedding(nn.Module):
    def __init__(self, embed_tokens, embed_positions, embed_scale, layer_norm):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.embed_positions = embed_positions
        self.embed_scale = embed_scale
        self.layer_norm= layer_norm

    def forward(self, src_tokens):
        x = self.embed_scale * self.embed_tokens(src_tokens)
        x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=0.3)
        x = x.transpose(0, 1)
        x = self.layer_norm(x)

        return x

class Linear(nn.Module):
    def __init__(self, embed_out):
        super().__init__()
        self.embed_out = embed_out

    def forward(self, x):
        x = x.transpose(0, 1)
        x = F.linear(x, self.embed_out)
        return x

def transfer_model(model):
    module = nn.ModuleList([])
    module.append(
        Token_Embedding(
            model.encoder.embed_tokens,
            model.encoder.embed_positions,
            model.encoder.embed_scale,
            model.encoder.layer_norm
        )
    )
    module.append(
        Token_Embedding(
            model.decoder.embed_tokens,
            model.decoder.embed_positions,
            model.decoder.embed_scale,
            model.decoder.layer_norm
        )
    )
    module.extend(model.encoder.layers)
    module.extend(model.decoder.layers)
    module.append(Linear(model.decoder.embed_out))
    return module

def forward_cuda_hook(*args):
    module = args[0]
    #print("to cuda", type(module))

    for p in module.parameters():
        #p.data = p.data.cuda(non_blocking=True)
        p.data = torch.randn(p.shape_, dtype=p.dtype_, device=torch.device('cuda'))

def forward_cpu_hook(*args):
    module = args[0]
    for p in module.parameters():
        ## if the first forward, not recompute
        if p.grad is None:
            # p.data = p.data.cpu()
            #p.grad = p.grad.cpu()
            p.data = torch.empty(0)


def backward_cpu_hook(*args):
    module = args[0]
    for p in module.parameters():
      
        p.data = torch.empty(0)
        #p.grad = torch.empty(0)


class Stage(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.start = 0
        self.end = len(module) - 1
        self.idx = [0 for i in range(len(module) - 3)]
        self.seq_id = 0

        self.dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def param_cpu(self):
        for m in self.module[2:-1]:
            for layer in m.layer:
                for p in layer.parameters():
                    if p.grad is not None:
                        p.data = p.data.cpu()

    def param_cuda(self):
        for m in self.module[2:-1]:
            for layer in m.layer:
                for p in layer.parameters():
                    if p.grad is not None:
                        p.data = p.data.cuda()

    def grad_cpu(self):
        for m in self.module[2:-1]:
            for layer in m.layer:
                for p in layer.parameters():
                    if p.grad is not None:
                        p.grad.data = p.grad.data.cpu()

    def cuda(self):
        for i in range(2, len(self.module) - 1):
            self.module[i].layer_norm.cuda()
        for i in range(self.start, self.end):
            if i > 1 and i < len(self.module) - 1:
                self.module[i].layer[self.idx[i - 2]].cuda()
                # self.module[i].layer_norm.cuda()
            else:
                self.module[i].cuda()
        return self
    # def cuda(self):
    #     for i in range(self.start, self.end):
    #         self.module[i].cuda()

    #     return self

    def forward(self, input0, input1):
        return cp.checkpoint(self.forward_, input0, input1, self.dummy)

    def forward_(self, input0, input1, dummy=None):
        start = self.start
        if self.start == 0:
            assert self.end > 1
            input0 = self.module[0](input0)
            input1 = self.module[1](input1)
            start = 2
        for i in range(start, self.end):
            m = self.module[i]
            if i > 1 and i < len(self.module) - 1:
                m.idx = self.idx[i - 2]
            if isinstance(m, TransformerEncoderLayer):
                input0 = m(input0)
            elif isinstance(m, TransformerDecoderLayer):
                input1 = m(input1, input0)
            else:
                assert isinstance(m, Linear)
                return m(input1)
        return input0, input1

def o():
    return TransformerModel.build_model(Args(), Task())

def eval_stage():
    model = TransformerModel.build_model(Args(), Task())
    module = transfer_model(model)
    return Stage(module)

def full_model(modules):
    return TRANSFORMERPartitioned(modules)

def model(criterion):
    model = TransformerModel.build_model(Args(), Task())
    module = transfer_model(model)
    for m in module[2:-1]:
        for layer in m.layer:
            for p in layer.parameters():
                p.shape_ = p.shape
                p.dtype_ = p.dtype
                p.data = torch.empty(0)
            layer.register_forward_pre_hook(forward_cuda_hook)
            layer.register_forward_hook(forward_cpu_hook)
            # layer.register_backward_pre_hook(to_cuda_hook)
            layer.register_backward_hook(backward_cpu_hook)

    stages = []
    p = [0, 17, 33, 50, 51]
    for i in range(4):
        stage = Stage(module)
        stage.start = p[i]
        stage.end = p[i + 1]
        stages.append(stage)
    return [
        (stages[0], ["input0", "input1"], ["out0", "out1"]),
        (stages[1], ["out0", "out1"], ["out2", "out3"]),
        (stages[2], ["out2", "out3"], ["out4", "out5"]),
        (stages[3], ["out4", "out5"], ["out6"]),
        (criterion, ["out6"], ["loss"])
    ]
