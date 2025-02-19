# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NVIDIA's Memory Transformer.
"""

import functools
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lpzero.common.utils import map_to_list
from lpzero.model.mem_transformer.mem_transformer_utils.log_uniform_sampler import LogUniformSampler, sample_logits
from lpzero.model.model_base import ArchaiModel
from lpzero.model.model_utils.adaptive_embedding import AdaptiveEmbedding
from lpzero.model.model_utils.primer_ez import DWiseConvPrimerEZ, PositionwiseFFPrimerEZ
from lpzero.model.model_utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax


@torch.jit.script
def add_and_scale(tensor1, tensor2, alpha: float):
    return alpha * (tensor1 + tensor2)


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = self.CoreNet(inp)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, use_cache=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.use_cache = use_cache

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None, past_key_values=None):
        # multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            # layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        if past_key_values is not None:
            past_k, past_v = torch.unbind(past_key_values, dim=0)
            head_k = torch.cat((past_k, head_k), dim=0)
            head_v = torch.cat((past_v, head_v), dim=0)

        if self.use_cache:
            present_key_values = torch.stack([head_k, head_v], dim=0)
        else:
            present_key_values = None

        # [bsz x n_head x qlen x klen]
        attn_score = torch.einsum('ibnd,jbnd->bnij', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, None, :, :], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, None, :, :], -float('inf'))

        # [bsz x qlen x klen x n_head]
        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropatt(attn_prob)

        # [bsz x n_head x qlen x klen] * [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('bnij,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output, present_key_values


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False,
                 primer_ez=False, use_cache=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.primer_ez = primer_ez
        self.use_cache = use_cache

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        if self.primer_ez:
            self.dconv = DWiseConvPrimerEZ(self.d_model)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask.bool()
        else:
            return mask.flip(0).bool()

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                   device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:, :, None, None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), x.size(1), x.size(2), 1),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=3)

        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))

        x = x_padded.narrow(2, 1, x_padded.size(2) - 1).view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)
        self.flops = 0

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None, past_key_values=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        self.flops = 0
        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            self.flops += torch.prod(torch.tensor(w_heads.size())) * self.d_model
            
            r_head_k = self.r_net(r)
            self.flops += torch.prod(torch.tensor(r_head_k.size())) * self.d_model

            if self.primer_ez:
                w_heads = self.dconv(w_heads)
                self.flops += torch.prod(torch.tensor(w_heads.size())) * self.dconv.kernel_size * self.dconv.kernel_size

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            self.flops += torch.prod(torch.tensor(w_heads.size())) * self.d_model

            r_head_k = self.r_net(r)
            self.flops += torch.prod(torch.tensor(r_head_k.size())) * self.d_model

            if self.primer_ez:
                w_heads = self.dconv(w_heads)
                self.flops += torch.prod(torch.tensor(w_heads.size())) * self.dconv.kernel_size * self.dconv.kernel_size

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # klen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # klen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # qlen x n_head x d_head

        if past_key_values is not None:
            past_k, past_v, past_r = torch.unbind(past_key_values, dim=0)
            past_r = past_r[:, 0, :, :]
            w_head_k = torch.cat((past_k, w_head_k), dim=0)
            w_head_v = torch.cat((past_v, w_head_v), dim=0)
            r_head_k = torch.cat((r_head_k, past_r), dim=0)

        if self.use_cache:
            # Caveat that allows for torch `stack` and `unbind`
            _r_head_k = r_head_k.unsqueeze(1).expand(-1, bsz, -1, -1)
            present_key_values = torch.stack([w_head_k, w_head_v, _r_head_k], dim=0)
        else:
            present_key_values = None

        # compute attention score
        rw_head_q = w_head_q + r_w_bias                                # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->bnij', (rw_head_q, w_head_k))    # bsz x n_head x qlen x klen
        self.flops += torch.prod(torch.tensor(AC.size())) * w_head_k.size(-1)

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->bnij', (rr_head_q, r_head_k))     # bsz x n_head x qlen x klen
        self.flops += torch.prod(torch.tensor(BD.size())) * r_head_k.size(-1)
        BD = self._rel_shift(BD)

        # [bsz x n_head x qlen x klen]
        attn_score = add_and_scale(AC, BD, self.scale)

        # compute attention probability
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, None, :, :], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, None, :, :], -float('inf'))

        # [bsz x n_head x qlen x klen]
        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum('bnij,jbnd->ibnd', (attn_prob, w_head_v))
        self.flops += torch.prod(torch.tensor(attn_vec.size())) * attn_prob.size(-1)

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        self.flops += torch.prod(torch.tensor(attn_out.size())) * attn_vec.size(-1)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output, present_key_values


class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None, past_key_values=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)

            if self.primer_ez:
                w_heads = self.dconv(w_heads)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)

            if self.primer_ez:
                w_heads = self.dconv(w_heads)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)
        plen = 0

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if past_key_values is not None:
            past_k, past_v = torch.unbind(past_key_values, dim=0)
            plen = past_k.size(0)
            w_head_k = torch.cat((past_k, w_head_k), dim=0)
            w_head_v = torch.cat((past_v, w_head_v), dim=0)

        if self.use_cache:
            present_key_values = torch.stack([w_head_k, w_head_v], dim=0)
        else:
            present_key_values = None

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen+plen-r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen+plen-r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-(klen+plen):]
            r_bias = r_bias[-(klen+plen):]

        r_bias = r_bias.t()

        # compute attention score
        rw_head_q = w_head_q + r_w_bias[None]                        # qlen x bsz x n_head x d_head

        AC = torch.einsum('ibnd,jbnd->bnij', (rw_head_q, w_head_k))  # bsz x n_head x qlen x klen
        B_ = torch.einsum('ibnd,jnd->bnij', (w_head_q, r_emb))       # bsz x n_head x qlen x klen
        D_ = r_bias[None, :, None, :]                                # 1   x n_head x    1 x klen
        BD = self._rel_shift(B_ + D_)

        # [bsz x qlen x klen x n_head]
        attn_score = add_and_scale(AC, BD, self.scale)

        # compute attention probability
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, None, :, :], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, None, :, :], -float('inf'))

        # [bsz x n_head x qlen x klen]
        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum('bnij,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output, present_key_values


class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, use_cache=False, **kwargs):
        super(DecoderLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, use_cache=use_cache, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, dec_attn_mask=None, mems=None, past_key_values=None):

        output, present_key_values = self.dec_attn(dec_inp, attn_mask=dec_attn_mask,
                                                   mems=mems, past_key_values=past_key_values)
        output = self.pos_ff(output)

        return output, present_key_values


class RelLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 primer_conv=False, primer_square=False, use_cache=False, **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head,
                                                  dropout, primer_ez=primer_conv, use_cache=use_cache,
                                                  **kwargs)

        if primer_square:
            self.pos_ff = PositionwiseFFPrimerEZ(d_model, d_inner, dropout,
                                                 pre_lnorm=kwargs.get('pre_lnorm'))
        else:
            self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                         pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None, mems=None, past_key_values=None):

        output, present_key_values = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias,
                                                   attn_mask=dec_attn_mask,
                                                   mems=mems, past_key_values=past_key_values)
        output = self.pos_ff(output)

        return output, present_key_values


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 primer_conv=False, primer_square=False, use_cache=False, **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                                                         d_head, dropout,
                                                         primer_ez=primer_conv, use_cache=use_cache,
                                                         **kwargs)

        if primer_square:
            self.pos_ff = PositionwiseFFPrimerEZ(d_model, d_inner, dropout,
                                                 pre_lnorm=kwargs.get('pre_lnorm'))
        else:
            self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                         pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None, past_key_values=None):

        output, present_key_values = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                                                   attn_mask=dec_attn_mask,
                                                   mems=mems, past_key_values=past_key_values)
        output = self.pos_ff(output)

        return output, present_key_values


class MemTransformerLM(ArchaiModel):
    def __init__(self, n_token, n_layer=16, n_head=8, d_model=512, d_head=64, d_inner=2048,
                 dropout=0.1, dropatt=0.0, dtype=None, tie_weight=True, d_embed=512,
                 div_val=1, tie_projs=None, pre_lnorm=False,
                 tgt_len=192, ext_len=0, mem_len=192,
                 cutoffs=None, adaptive=False,
                 same_length=False, attn_type=0, clamp_len=-1, sample_softmax=-1,
                 weight_init_type='normal', weight_init_range=0.1, weight_init_std=0.02,
                 proj_init_std=0.01, init_std=0.02,
                 primer_conv=False, primer_square=False, use_cache=False, **kwargs):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token # number of tokens in vocab

        d_embed = d_model if d_embed < 0 else d_embed
        d_inner = map_to_list(d_inner, n_layer)
        n_head = map_to_list(n_head, n_layer)
        d_head = [d_model // n_h for n_h in n_head] if d_head < 0 else map_to_list(d_head, n_layer)

        assert len(d_inner) == n_layer and len(n_head) == n_layer and len(d_head) == n_layer

        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs,
                                          div_val=div_val)

        self.drop = nn.Dropout(dropout)

        self.tie_weight = tie_weight
        self.tie_projs = tie_projs
        self.div_val = div_val

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len # extended context length, default is 0
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type
        self.use_cache = use_cache

        self.layers = nn.ModuleList()
        # the default attention
        if attn_type == 0:
            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        n_head[i], d_model, d_head[i], d_inner[i], dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm, primer_conv=primer_conv,
                        primer_square=primer_square, use_cache=use_cache)
                )
        # learnable embeddings
        elif attn_type == 1:
            for i in range(n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        n_head[i], d_model, d_head[i], d_inner[i], dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm, primer_conv=primer_conv,
                        primer_square=primer_square, use_cache=use_cache)
                )
        # absolute embeddings
        elif attn_type in [2, 3]:
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head[i], d_model, d_head[i], d_inner[i], dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm, use_cache=use_cache)
                )

        self.sample_softmax = sample_softmax
        # use sampled softmax
        if sample_softmax > 0: # default is to not use
            self.out_layer = nn.Linear(d_model, n_token)
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)

        # use adaptive softmax (including standard softmax)
        else:
            if tie_weight: # default is True
                # word_emb.emb_layers has only one element with embedding weight matrix
                emb_layers = [i.weight for i in self.word_emb.emb_layers]
            else:
                emb_layers = None

            emb_projs = self.word_emb.emb_projs # nn.ParameterList of len=0

            self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model,
                                                    cutoffs, adaptive,
                                                    div_val=div_val,
                                                    tie_projs=tie_projs,
                                                    out_projs=emb_projs,
                                                    out_layers_weights=emb_layers)


        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

        # initialize weights
        weight_init_params = {
            'weight_init_type': weight_init_type,
            'weight_init_range': weight_init_range,
            'weight_init_std': weight_init_std,
            'proj_init_std': proj_init_std,
        }

        self.apply(functools.partial(weights_init, **weight_init_params))
        # ensure embedding init is not overridden by out_layer in case of weight sharing
        self.word_emb.apply(functools.partial(weights_init, **weight_init_params))

    def get_params(self):
        params = {}

        params['embedding'] = self.get_params_from_layer(['AdaptiveEmbedding'])
        params['softmax'] = self.get_params_from_layer(['ProjectedAdaptiveLogSoftmax'])
        params['attention'] = self.get_params_from_layer(['MultiHeadAttn', 'RelPartialLearnableMultiHeadAttn', 'RelLearnableMultiHeadAttn'])
        params['ff'] = self.get_params_from_layer(['Sequential'])
        params['layer_norm'] = self.get_params_from_layer(['LayerNorm'])

        params['non_embedding'] = params['attention'] + params['ff'] + params['layer_norm']
        params['total'] = params['non_embedding'] + params['embedding'] + params['softmax']

        return params

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        # default attention
        if self.attn_type == 0:
            self.pos_emb = PositionalEmbedding(self.d_model)
            # Sets learnable attributes per layer due to possible different sizes of `n_head` and `d_head`
            # yet the added extra parameters have a negligible training cost (+0.47%)
            for i, _ in enumerate(self.layers):
                setattr(self, f'r_w_bias_{i}', nn.Parameter(torch.Tensor(self.n_head[i], self.d_head[i]).zero_()))
                setattr(self, f'r_r_bias_{i}', nn.Parameter(torch.Tensor(self.n_head[i], self.d_head[i]).zero_()))
        # learnable
        elif self.attn_type == 1:
            # Sets learnable attributes per layer due to possible different sizes of `n_head` and `d_head`
            # yet the added extra parameters have a negligible training cost (+0.69%)
            for i, _ in enumerate(self.layers):
                setattr(self, f'r_emb_{i}', nn.Parameter(torch.Tensor(self.max_klen, self.n_head[i], self.d_head[i]).zero_()))
                setattr(self, f'r_w_bias_{i}', nn.Parameter(torch.Tensor(self.n_head[i], self.d_head[i]).zero_()))
                setattr(self, f'r_bias_{i}', nn.Parameter(torch.Tensor(self.max_klen, self.n_head[i]).zero_()))
        # absolute standard
        elif self.attn_type == 2:
            self.pos_emb = PositionalEmbedding(self.d_model)
        # absolute deeper SA
        elif self.attn_type == 3:
            self.r_emb = nn.Parameter(torch.Tensor(self.n_layer, self.max_klen, self.d_model).zero_())

    def reset_length(self, tgt_len, ext_len, mem_len):
        if tgt_len < 1:
            raise RuntimeError(f'tgt_len should be >= 1, but got {tgt_len}')
        if ext_len < 0:
            raise RuntimeError(f'ext_len should be >= 0, but got {ext_len}')
        if mem_len < 0:
            raise RuntimeError(f'mem_len should be >= 0, but got {mem_len}')
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            param = next(self.parameters())
            mems = torch.empty(self.n_layer, 0, dtype=param.dtype,
                               device=param.device)
            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            stacked = torch.stack(hids)
            if (
                self.mem_len == self.tgt_len
                and self.ext_len == 0
                and stacked.size(1) == self.mem_len
            ):
                new_mems = stacked.detach()
            else:
                end_idx = mlen + max(0, qlen - self.ext_len)
                beg_idx = max(0, end_idx - self.mem_len)
                if mems.numel():
                    cat = torch.cat([mems, stacked], dim=1)
                else:
                    cat = stacked
                new_mems = cat[:, beg_idx:end_idx].detach()

        return new_mems

    def _forward(self, dec_inp, mems=None, past_key_values=None):
        qlen, bsz = dec_inp.size()

        word_emb = self.word_emb(dec_inp)

        mlen = mems[0].size(0) if mems is not None else 0
        plen = past_key_values[0][0].size(0) if past_key_values[0] is not None else 0
        klen = mlen + qlen
        # `plen` should be taken into account when creating the
        # attention mask because `past_key_values` might be used
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen+plen)
            mask_len = klen - self.mem_len - 1
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen+plen)
                             + torch.tril(all_ones, -mask_shift_len)).bool()
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen+plen), diagonal=1+mlen+plen).bool()

        hids = []
        pasts_key_values = ()
        # default
        if self.attn_type == 0:
            pos_seq = torch.arange(klen+plen-1, plen-1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)

            for i, (layer, past_key_values_i) in enumerate(zip(self.layers, past_key_values)):
                hids.append(core_out.detach())
                mems_i = None if mems is None else mems[i]
                core_out, past_key_values_i = layer(core_out, pos_emb, getattr(self, f'r_w_bias_{i}'),
                                                    getattr(self, f'r_r_bias_{i}'), dec_attn_mask=dec_attn_mask,
                                                    mems=mems_i, past_key_values=past_key_values_i)
                pasts_key_values = pasts_key_values + (past_key_values_i, )
        # learnable
        elif self.attn_type == 1:
            core_out = self.drop(word_emb)
            for i, (layer, past_key_values_i) in enumerate(zip(self.layers, past_key_values)):
                hids.append(core_out.detach())
                if self.clamp_len > 0:
                    r_emb = getattr(self, f'r_emb_{i}')[-self.clamp_len:]
                    r_bias = getattr(self, f'r_bias_{i}')[-self.clamp_len:]
                else:
                    r_emb, r_bias = getattr(self, f'r_emb_{i}'), getattr(self, f'r_bias_{i}')

                mems_i = None if mems is None else mems[i]
                core_out, past_key_values_i = layer(core_out, r_emb, getattr(self, f'r_w_bias_{i}'),
                                                    r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i,
                                                    past_key_values=past_key_values_i)
                pasts_key_values = pasts_key_values + (past_key_values_i, )
        # absolute
        elif self.attn_type == 2:
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb + pos_emb[-qlen:])

            for i, (layer, past_key_values_i) in enumerate(zip(self.layers, past_key_values)):
                hids.append(core_out.detach())
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and len(mems_i) and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out, past_key_values_i = layer(core_out, dec_attn_mask=dec_attn_mask,
                                                    mems=mems_i, past_key_values=past_key_values_i)
                pasts_key_values = pasts_key_values + (past_key_values_i, )
        elif self.attn_type == 3:
            core_out = self.drop(word_emb)

            for i, (layer, past_key_values_i) in enumerate(zip(self.layers, past_key_values)):
                hids.append(core_out.detach())
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and len(mems_i) and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(mlen-cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                core_out, past_key_values_i = layer(core_out, dec_attn_mask=dec_attn_mask,
                                                    mems=mems_i, past_key_values=past_key_values_i)
                pasts_key_values = pasts_key_values + (past_key_values_i, )

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, qlen, mlen)

        return core_out, new_mems, pasts_key_values

    def forward(self, input_ids:torch.Tensor, labels:Optional[torch.Tensor]=None, mems:Optional[torch.Tensor]=None,
                past_key_values:Optional[torch.Tensor]=None, output_loss=True, output_prediction_scores=False):
        # input_ids and labels are transposed within the code to avoid major changes
        # input_ids -> [seq_len, batch_size], labels -> [seq_len, batch_size]
        # Returns:
        # loss -> [batch_size, seq_len], prediction_scores -> [batch_size, seq_len, vocab_size]
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.

        # Transposes `input_ids` and `labels` to seq_len x batch_size
        input_ids = input_ids.t()
        if labels is not None:
            labels = labels.t()

        if mems is None:
            mems = self.init_mems()

        if labels is None:
            output_loss = False

        if past_key_values is None:
            past_key_values = tuple([None] * self.n_layer)

        hidden, mems, past_key_values = self._forward(input_ids, mems=mems, past_key_values=past_key_values)

        tgt_len = labels.size(0) if labels is not None else input_ids.size(0)

        pred_hid = hidden[-tgt_len:]
        if self.sample_softmax > 0 and self.training:
            raise NotImplementedError('Computing log probabilities is not implemented for sample_softmax mode')
            assert self.tie_weight
            logit = sample_logits(self.word_emb, self.out_layer.bias, labels,
                                  pred_hid, self.sampler)
            loss = -F.log_softmax(logit, -1)[:, :, 0]
        else:
            # As we are transposing the `labels`, we need it in a contiguous
            # piece of memory before applying a tensor visualization (view)
            loss, prediction_scores = self.crit(hidden=pred_hid.view(-1, pred_hid.size(-1)),
                                                target=labels.contiguous().view(-1) if labels is not None else None,
                                                output_loss=output_loss, output_prediction_scores=output_prediction_scores)
            # loss -> [batch_size, tgt_len]
            # prediction_scores -> [batch_size, tgt_len, vocab_size]
            loss = loss.view(-1, tgt_len) if labels is not None else None
            prediction_scores = prediction_scores.view(input_ids.size(1), tgt_len, -1) if prediction_scores is not None else None

        return (loss, prediction_scores, mems, past_key_values)

def init_weight(weight, weight_init_type:str, weight_init_range:float, weight_init_std:float):
    """Intialize given parameters using specified strategy"""
    if weight_init_type == 'uniform':
        nn.init.uniform_(weight, -weight_init_range, weight_init_range)
    elif weight_init_type == 'normal': # default
        nn.init.normal_(weight, 0.0, weight_init_std)

def init_bias(bias):
    nn.init.constant_(bias, 0.0)

def weights_init(m, weight_init_type:str, weight_init_range:float, weight_init_std:float, proj_init_std:float):
    """Initialize weights of module using specified strategy"""
    classname = m.__class__.__name__

    weight_init_params = {
        'weight_init_type': weight_init_type,
        'weight_init_range': weight_init_range,
        'weight_init_std': weight_init_std
    }

    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight, **weight_init_params)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight, **weight_init_params)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight, **weight_init_params)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, proj_init_std)
        if hasattr(m, 'out_layers_weights'):
            for i in range(len(m.out_layers_weights)):
                if m.out_layers_weights[i] is not None:
                    init_weight(m.out_layers_weights[i], **weight_init_params)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, weight_init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        for i in range(m.n_layer):
            if hasattr(m, f'r_emb_{i}'):
                init_weight(getattr(m, f'r_emb_{i}'), **weight_init_params)
            if hasattr(m, f'r_w_bias_{i}'):
                init_weight(getattr(m, f'r_w_bias_{i}'), **weight_init_params)
            if hasattr(m, f'r_r_bias_{i}'):
                init_weight(getattr(m, f'r_r_bias_{i}'), **weight_init_params)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='unit test')

    parser.add_argument('--n_layer', type=int, default=16, help='')
    parser.add_argument('--n_token', type=int, default=267735, help='') # 267735
    parser.add_argument('--n_head', type=int, default=8, help='')
    parser.add_argument('--d_head', type=int, default=64, help='')
    parser.add_argument('--d_model', type=int, default=512, help='')
    parser.add_argument('--d_embed', type=int, default=512, help='')
    parser.add_argument('--d_inner', type=int, default=2048, help='')
    parser.add_argument('--div_val', type=int, default=1, help='') # Dividend value for adaptive input and softmax
    parser.add_argument('--dropout', type=float, default=0.1, help='')
    parser.add_argument('--cuda', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--multi_gpu', action='store_true', help='')

    args = parser.parse_args()

    tgt_len, mem_len, ext_len = 192, 192, 0
    cutoffs =  [args.n_token//8, args.n_token//4, args.n_token // 2] # [19997, 39997, 199997]
    tie_projs = [False] + [True] * len(cutoffs)

    model = MemTransformerLM(args.n_token, args.n_layer, args.n_head,
                                args.d_model, args.d_head, args.d_inner,
                                args.dropout, dropatt=args.dropout,
                                tie_weight=True, d_embed=args.d_embed,
                                div_val=args.div_val, tie_projs=tie_projs,
                                pre_lnorm=True, tgt_len=tgt_len,
                                ext_len=ext_len, mem_len=mem_len,
                                cutoffs=cutoffs, attn_type=0,
                                dtype=None)

    print('# total params', sum(p.numel() for p in model.parameters()))
    print('# embd params', sum(p.numel() for p in model.word_emb.parameters()))
    print('# layer params', sum(p.numel() for p in model.layers[0].parameters()))
