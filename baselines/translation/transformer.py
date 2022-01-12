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

from apex.normalization.fused_layer_norm import FusedLayerNorm

from layers import get_layer


def t2t_architecture(args):
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.dropout = getattr(args, 'dropout', 0.3)

    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
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
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024
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

        self.layer_norm = FusedLayerNorm(embed_dim)

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

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = checkpoint(layer.forward, x, encoder_padding_mask)

        x = self.layer_norm(x)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

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

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, left_pad=False):
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
            TransformerDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])
        self.adaptive_softmax = None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary), args.decoder_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.dropout
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=embed_dim ** -0.5)
        self.layer_norm = FusedLayerNorm(embed_dim)

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
        # The tensor needs to copy transposed because
        # fused dropout is not capable of handing strided data
        x = x.transpose(0, 1)

        # decoder layers
        for layer in self.layers:
            x = checkpoint(layer.forward,
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
            )

        x = self.layer_norm(x)

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
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = nn.MultiheadAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([FusedLayerNorm(self.embed_dim) for i in range(2)])


    def forward(self, x, encoder_padding_mask):
        residual = x

        x = self.layer_norms[0](x)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.layer_norms[1](x)

        x = F.threshold(self.fc1(x),0,0)
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block."""

    def __init__(self, args, no_encoder_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = nn.MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout

        self.self_attn_layer_norm = FusedLayerNorm(self.embed_dim)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = nn.MultiheadAttention(
                self.embed_dim, args.decoder_attention_heads,
                dropout=args.attention_dropout,
            )
            self.encoder_attn_layer_norm = FusedLayerNorm(self.embed_dim)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = FusedLayerNorm(self.embed_dim)


    def forward(self, x, encoder_out, encoder_padding_mask):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        if self.encoder_attn is not None:
            residual = x
            x = self.encoder_attn_layer_norm(x)
            x, _ = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = F.threshold(self.fc1(x),0,0)
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        return x


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LayerNorm(embedding_dim):
    m = FusedLayerNorm(embedding_dim)
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
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, num_embeddings + padding_idx + 1)
    return m