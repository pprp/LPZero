# ============================================
__author__ = 'Sachin Mehta'
__maintainer__ = 'Sachin Mehta'
# ============================================

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.delight_modules import DEFAULT_MIN_DEXTRA_LAYERS, DEFAULT_WIDTH_MULTIPLIER
from fairseq.delight_modules.activation_layers import get_activation_layer
from fairseq.delight_modules.dextra_unit import DExTraUnit
from fairseq.delight_modules.nn_functions import get_weight_layer
from fairseq.delight_modules.normalization_layers import get_norm_layer
from fairseq.modules import SingleHeadAttention
from torch import Tensor


class DeLighTTransformerEncoderLayer(nn.Module):
    """DeLight Encoder layer"""

    def __init__(
        self,
        args,
        embed_dim,
        width_multiplier=DEFAULT_WIDTH_MULTIPLIER,
        dextra_depth=DEFAULT_MIN_DEXTRA_LAYERS,
        dextra_proj=2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        assert embed_dim % dextra_proj == 0

        self.proj_dim = embed_dim // dextra_proj
        self.dextra_layer = DExTraUnit(
            in_features=self.embed_dim,
            in_proj_features=self.proj_dim,
            out_features=self.proj_dim,
            width_multiplier=width_multiplier,
            dextra_depth=dextra_depth,
            dextra_dropout=args.delight_dropout,
            max_glt_groups=args.delight_enc_max_groups,
            act_type=args.act_type,
            use_bias=True,
            norm_type=args.norm_type,
            glt_shuffle=args.glt_shuffle,
            is_iclr_version=args.define_iclr,
        )

        self.self_attn = SingleHeadAttention(
            q_in_dim=self.proj_dim,
            kv_in_dim=self.proj_dim,
            proj_dim=self.proj_dim,
            out_dim=self.embed_dim,
            dropout=args.attention_dropout,
            bias=True,
            self_attention=True,
            encoder_decoder_attention=False,
        )

        self.self_attn_layer_norm = get_norm_layer(
            name=args.norm_type, out_features=self.embed_dim
        )
        self.dropout = args.dropout
        self.norm_fn = args.norm_type
        self.act_type = args.act_type
        self.activation_fn = get_activation_layer(name=args.act_type)
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.encoder_normalize_before

        # Light-weight FFN
        self.ffn_dropout = args.ffn_dropout
        ffn_red_factor = args.delight_enc_ffn_red
        assert (
            self.embed_dim % ffn_red_factor == 0
        ), '{}/{} should be a perfect divisor'.format(self.embed_dim, ffn_red_factor)
        light_ffn_dim = self.embed_dim // ffn_red_factor
        self.fc1 = get_weight_layer(
            name='linear',
            in_features=self.embed_dim,
            out_features=light_ffn_dim,
            use_bias=True,
        )
        self.fc2 = get_weight_layer(
            name='linear',
            in_features=light_ffn_dim,
            out_features=self.embed_dim,
            use_bias=True,
        )

        self.final_layer_norm = get_norm_layer(
            name=args.norm_type, out_features=self.embed_dim
        )

    def __repr__(self):
        s = (
            '{name}(in_features={embed_dim}, out_features={embed_dim}, dropout={dropout},'
            'activation_dropout={activation_dropout}, ffn_dropout={ffn_dropout}, '
            'activation_fn={act_type}, norm_fn={norm_fn})'
        )
        s += '\n \t Dextra Layer: \n \t \t {}'.format(self.dextra_layer)
        s += '\n \t Self Attention: \n \t \t {}'.format(self.self_attn)
        s += '\n \t     Light-weight FFN: \n \t     |---- {} \n \t     |---- {}'.format(
            self.fc1, self.fc2
        )
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {'0': 'self_attn_layer_norm', '1': 'final_layer_norm'}
        for old, new in layer_norm_map.items():
            for m in ('weight', 'bias'):
                k = '{}.layer_norms.{}.{}'.format(name, old, m)
                if k in state_dict:
                    state_dict['{}.{}.{}'.format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
            T_tgt is the length of query, while T_src is the length of key,
            though here both query and key is x here,
            attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        x = self.dextra_layer(x)

        x, _ = self.self_attn(
            query=x,
            key_value=None,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Light-weight FFN
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=float(self.activation_dropout),
                      training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.ffn_dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x

    def compute_macs_params(self, S=1):
        macs = 0
        n_params = 0
        macs_attn = 0

        # Layer Norms
        # MACS are zero for LayerNorm because they can be fused
        n_params += sum([p.numel()
                        for p in self.self_attn_layer_norm.parameters()])

        # Dextra layer
        dextra_layer = self.dextra_layer.compute_macs_params()
        n_params += dextra_layer['params']
        macs += dextra_layer['macs'] * S

        # Attn
        self_attn_layer = self.self_attn.compute_macs_params(T=S, S=S)
        macs += self_attn_layer['macs']
        n_params += self_attn_layer['params']
        macs_attn += self_attn_layer['macs_attn']

        # FFN
        fc1_layer = self.fc1.compute_macs_params()
        # scale MACS by S because S tokens can be processed in parallel
        macs += fc1_layer['macs'] * S
        n_params += fc1_layer['params']

        fc2_layer = self.fc2.compute_macs_params()
        # scale MACS by S because S tokens can be processed in parallel
        macs += fc2_layer['macs'] * S
        n_params += fc2_layer['params']

        n_params += sum([p.numel()
                        for p in self.final_layer_norm.parameters()])

        return {
            'name': self.__class__.__name__,
            'macs': macs,
            'params': n_params,
            'macs_attn': macs_attn,
        }


if __name__ == '__main__':
    pass
