import torch
import torch.nn as nn
import torch.nn.functional as F


SEED = 123 # or whatever you choose
# random.seed(SEED) # if you're using random
# np.random.seed(123) # if you're using numpy
torch.manual_seed(SEED) # torch.cuda.manual_seed_all(SEED) is not required
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class TRANSFORMERPartitioned(torch.nn.Module):
    def __init__(self, modules):
        super(TRANSFORMERPartitioned, self).__init__()

        self.encoder = Encoder(nn.ModuleList([modules[0]] + modules[2:26]))
        self.decoder = Decoder(nn.ModuleList([modules[1]] + modules[26:]))

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        return decoder_out

    def max_positions(self):
        """Maximum length supported by the model."""
        return (1024, 1024)

    def make_generation_fast_(self, **kwargs):
        # remove weight norm from all modules in the network
        def apply_remove_weight_norm(module):
            try:
                nn.utils.remove_weight_norm(module)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(apply_remove_weight_norm)

        def apply_make_generation_fast_(module):
            if module != self and hasattr(module, 'make_generation_fast_'):
                module.make_generation_fast_(**kwargs)

        self.apply(apply_make_generation_fast_)

        def train(mode):
            if mode:
                raise RuntimeError('cannot train after make_generation_fast')

        # this model should no longer be used for training
        self.eval()

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return 1024

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""

        #logits = net_output[0].float()
        logits = net_output[0] #.float()
        if log_probs:
            return F.log_softmax(logits, dim=-1, dtype=torch.float32)
        else:
            return F.softmax(logits, dim=-1, dtype=torch.float32)

    def cuda(self):
        self.encoder.cuda()
        self.decoder.cuda()
        return self

class Encoder(torch.nn.Module):
    def __init__(self, module):
        super(Encoder, self).__init__()

        self.module = module

    def forward(self, src_tokens, src_lengths):
        encoder_padding_mask = src_tokens.eq(1)
        x = self.module[0](src_tokens)
        for m in self.module[1:]:
            x = m(x)
        return {'encoder_out': x, 'encoder_padding_mask': encoder_padding_mask}

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def cuda(self):
        self.module[0].cuda()
        for m in self.module[1:]:
            m.layer[m.idx].cuda()
            m.layer_norm.cuda()
        return self

class Decoder(torch.nn.Module):
    def __init__(self, module):
        super(Decoder, self).__init__()

        self.module = module

    def forward(self, prev_output_tokens, encoder_out):
        out = encoder_out['encoder_out']
        x = self.module[0](prev_output_tokens)
        for m in self.module[1:-1]:
            x = m(x, out)
        x = self.module[-1](x)
        return x, None
    
    def cuda(self):
        self.module[0].cuda()
        self.module[-1].cuda()
        for m in self.module[1:-1]:
            m.layer[m.idx].cuda()
            m.layer_norm.cuda()
        return self