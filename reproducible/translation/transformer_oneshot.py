import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from fairseq import options
from fairseq import utils

from fairseq.modules import (
    AdaptiveSoftmax, LearnedPositionalEmbedding, SinusoidalPositionalEmbedding
)

#from apex.normalization.fused_layer_norm import FusedLayerNorm

SEED = 123 # or whatever you choose
# random.seed(SEED) # if you're using random
# np.random.seed(123) # if you're using numpy
torch.manual_seed(SEED) # torch.cuda.manual_seed_all(SEED) is not required
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def t2t_architecture(args):
    args.attention_dropout = getattr(args, 'attention_dropout', 0)
    args.relu_dropout = getattr(args, 'relu_dropout', 0)

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.dropout = getattr(args, 'dropout', 0)

    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)


class TransformerModel(nn.Module):
    """Base class for encoder-decoder models."""

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        return decoder_out

    def get_targets(self, sample):
        """Get targets from either the sample or the net's output."""
        return sample['target']

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, 'adaptive_softmax') and self.adaptive_softmax is not None:
            assert sample is not None and 'target' in sample
            out = self.adaptive_softmax.get_log_prob(net_output, sample['target'])
            return out.exp_() if not log_probs else out

        logits = net_output #.float()
        if log_probs:
            return F.log_softmax(logits, dim=-1, dtype=torch.float32)
        else:
            return F.softmax(logits, dim=-1, dtype=torch.float32)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        t2t_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 512
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 512
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        encoder_embed_tokens = build_embedding(
            src_dict, args.encoder_embed_dim, args.encoder_embed_path
        )
        decoder_embed_tokens = build_embedding(
            tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
        )

        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
        decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)
        return TransformerModel(encoder, decoder)


class TransformerEncoder(nn.Module):
    """Transformer encoder."""

    def __init__(self, args, dictionary, embed_tokens, left_pad=True):
        super().__init__()
        self.dictionary = dictionary
        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])

        self.layer_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, src_tokens, src_lengths):
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        # The tensor needs to copy transposed because
        # fused dropout is not capable of handing strided data
        x = x.transpose(0, 1)
        x = self.layer_norm(x)

        # encoder layers
        for layer in self.layers:
            x = layer(x)

        return x # T x B x C

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if 'encoder.embed_positions.weights' in state_dict:
                del state_dict['encoder.embed_positions.weights']
            state_dict['encoder.embed_positions._float_tensor'] = torch.FloatTensor(1)
        return state_dict


class TransformerDecoder(nn.Module):
    """Transformer decoder."""

    def __init__(self, args, dictionary, embed_tokens, left_pad=False):
        super().__init__()
        self.dictionary = dictionary
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        embed_dim = embed_tokens.embedding_dim
        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args)
            for _ in range(args.decoder_layers)
        ])
        self.adaptive_softmax = None

        # if args.adaptive_softmax_cutoff is not None:
        #     self.adaptive_softmax = AdaptiveSoftmax(
        #         len(dictionary), args.decoder_embed_dim,
        #         options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
        #         dropout=args.dropout
        #     )
        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=embed_dim ** -0.5)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, prev_output_tokens, encoder_out=None):
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
        ) if self.embed_positions is not None else None

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        x = self.layer_norm(x)

        # decoder layers
        for layer in self.layers:
            x = layer(
                x,
                encoder_out if encoder_out is not None else None,
            )

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = F.linear(x, self.embed_out)

        return x

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if 'decoder.embed_positions.weights' in state_dict:
                del state_dict['decoder.embed_positions.weights']
            state_dict['decoder.embed_positions._float_tensor'] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = 'decoder.layers.{}.layer_norms.{}.{}'.format(i, old, m)
                    if k in state_dict:
                        state_dict['decoder.layers.{}.{}.{}'.format(i, new, m)] = state_dict[k]
                        del state_dict[k]

        return state_dict


class TransformerEncoderLayer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.layer = encoder_layer(args)
        self.layer_norm = torch.nn.LayerNorm(args.encoder_embed_dim)
        self.idx = 0

    def forward(self, x):
        layer = self.layer[self.idx]
        if isinstance(layer, nn.MultiheadAttention):
            x, _ = layer(x, x, x)
        else:
            x = layer(x)
        #print("x type ", x)
        x = self.layer_norm(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block."""

    def __init__(self, args):
        super().__init__()
        self.layer = decoder_layer(args)
        self.layer_norm = torch.nn.LayerNorm(args.decoder_embed_dim)
        self.idx = 0

    def forward(self, x, encoder_out):
        layer = self.layer[self.idx]
        if isinstance(layer, nn.MultiheadAttention):
            if self.idx < len(self.layer) - 1:
                x, _ = layer(x, x, x)
            else:
                x, _ = layer(x, encoder_out, encoder_out)
        else:
            x = layer(x)
        x = self.layer_norm(x)
        return x


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LayerNorm(embedding_dim):
    m = torch.nn.LayerNorm(embedding_dim)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.bias, 0.)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings + padding_idx + 1, embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, num_embeddings + padding_idx)
    return m


def encoder_layer(args):
    ops = nn.ModuleList()
    embed_dim = args.encoder_embed_dim
    attention_dropout = 0 #args.attention_dropout
    # dropout = args.dropout
    # relu_dropout = args.relu_dropout

    # deterministic excution checklist:
    # conv ok  
    # multi head ok
    # gated linear ok

    # standard conv
    # For CV tasks
    # for ks in [3, 5, 7, 9]:
    #     for i in range(4):
    #         ops.append(
    #             StandardConv(embed_dim, embed_dim, ks)
    #         )

    # depthwise separable conv
    # For CV tasks
    # for ks in [3, 5, 7]:
    #     ops.append(
    #         SeparableConv(embed_dim, embed_dim, ks)
    #     )
    # lightweight conv
    # for ks in [3, 5, 7, 15]:
    #     for r in [1, 4, 16]:
    #         ops.append(
    #             LightweightConv(ks, r)
    #         )
    # h head attention
    # for NLP tasks
    for hs in [32, 32, 32, 32]:
        for i in range(24):
            ops.append(
                nn.MultiheadAttention(
                    embed_dim, hs,
                    dropout=attention_dropout,
                )
            )

    # gated linear
    ops.append(GatedLinear(embed_dim, embed_dim))
    # identity
    # ops.append(Identity())

    return ops


def decoder_layer(args):
    embed_dim = args.decoder_embed_dim
    attention_dropout = 0 #args.attention_dropout
    args.encoder_embed_dim = embed_dim
    ops = encoder_layer(args)
    ops.append(
        nn.MultiheadAttention(
            embed_dim, 32,
            dropout=attention_dropout,
        )
    )

    return ops


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def calc_padding(kernel_size):
    return (
        kernel_size // 2
        if kernel_size % 2 == 1
        else ((kernel_size - 1) // 2, kernel_size // 2)
    )


class StandardConv(nn.Module):
    """Standard convolutional layer class"""

    def __init__(self, input_depth, output_depth, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(input_depth, output_depth, kernel_size=kernel_size, padding=calc_padding(kernel_size))

    def forward(self, x):
        # T x B x C -> B x C x T
        x = x.permute(1, 2, 0)
        x = self.conv(x)
        # B x C x T -> T x B x C
        x = x.permute(2, 0, 1)
        return x


class SeparableConv(StandardConv):
    """Depthwise Seperable convolutional layer class"""

    def __init__(self, input_depth, output_depth, kernel_size):
        super().__init__(input_depth, output_depth, kernel_size)
        self.conv = nn.Sequential(
            nn.Conv1d(
                input_depth, input_depth, kernel_size=kernel_size,
                groups=input_depth, padding=calc_padding(kernel_size)
            ),
            nn.Conv1d(input_depth, output_depth, kernel_size=1)
        )


def unfold1d(x, kernel_size, padding_l, pad_value=0):
    """unfold T x B x C to T x B x C x K"""
    if kernel_size > 1:
        T, B, C = x.size()
        x = F.pad(
            x, (0, 0, 0, 0, padding_l, kernel_size - 1 - padding_l), value=pad_value
        )
        x = x.as_strided((T, B, C, kernel_size), (B * C, C, 1, B * C))
    else:
        x = x.unsqueeze(3)
    return x


class LightweightConv(nn.Module):
    """Lightweight Convolution assuming the input is TxBxC
    Args:
        kernel_size: convolution channels
        padding_l: padding to the left when using "same" padding
        num_heads: number of heads used. The weight is of shape (num_heads, 1, kernel_size)
        weight_dropout: the drop rate of the DropConnect to drop the weight
        weight_softmax: normalize the weight with softmax before the convolution
        bias: use bias
    Shape:
        Input: TxBxC, i.e. (timesteps, batch_size, input_size)
        Output: TxBxC, i.e. (timesteps, batch_size, input_size)
    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias:   the learnable bias of the module of shape `(input_size)`
    """

    def __init__(
        self,
        kernel_size=1,
        num_heads=1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.weight = nn.Parameter(torch.Tensor(num_heads, 1, kernel_size))
        nn.init.kaiming_uniform_(self.weight)
        self.padding_l = calc_padding(kernel_size)

    def forward(self, x, unfold=False):
        """Assuming the input, x, of the shape T x B x C and producing an output in the shape T x B x C
        args:
            x: Input of shape T x B x C, i.e. (timesteps, batch_size, input_size)
            unfold: unfold the input or not. If not, we use the matrix trick instead
        """
        if unfold:
            output = self._forward_unfolded(x)
        else:
            output = self._forward_expanded(x)

        return output

    def _forward_unfolded(self, x):
        """The conventional implementation of convolutions.
        Unfolding the input by having a window shifting to the right."""
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H

        weight = self.weight.view(H, K)
        # unfold the input: T x B x C --> T' x B x C x K
        x_unfold = unfold1d(x, self.kernel_size, self.padding_l, 0)
        x_unfold = x_unfold.view(T * B * H, R, K)

        weight = (
            weight.view(1, H, K).expand(T * B, H, K).contiguous().view(T * B * H, K, 1)
        )

        output = torch.bmm(x_unfold, weight)  # T*B*H x R x 1
        output = output.view(T, B, C)
        return output

    def _forward_expanded(self, x):
        """Turn the convolution filters into band matrices and do matrix multiplication.
        This is faster when the sequence is short, but less memory efficient.
        This is not used in the decoder during inference.
        """
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H

        weight = self.weight.view(H, K)
        weight = weight.view(1, H, K).expand(T * B, H, K).contiguous()
        weight = weight.view(T, B * H, K).transpose(0, 1)

        x = x.view(T, B * H, R).transpose(0, 1)
        P = self.padding_l
        if K > T and P == K - 1:
            weight = weight.narrow(2, K - T, T)
            K, P = T, T - 1
        # turn the convolution filters into band matrices
        weight_expanded = weight.new_zeros(B * H, T, T + K - 1, requires_grad=False)
        weight_expanded.as_strided((B * H, T, K), (T * (T + K - 1), T + K, 1)).copy_(
            weight
        )
        weight_expanded = weight_expanded.narrow(2, P, T)

        output = torch.bmm(weight_expanded, x)
        output = output.transpose(0, 1).contiguous().view(T, B, C)
        return output


class GatedLinear(nn.Module):
    def __init__(self, input_depth, output_depth):
        super().__init__()
        self.linear = nn.Linear(input_depth, 2 * output_depth)
        #nn.init.xavier_uniform_(self.linear.weight)
    def forward(self, x):
        return F.glu(self.linear(x))
