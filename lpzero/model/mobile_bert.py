import logging
import math

import datasets
import models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

from .cka import CKA_Minibatch_Grid
from .dynamic_ops import DynamicLinear
from .pca_torch import pca_torch

# device = torch.device('cuda:5')
# cuda_cka = CudaCKA(device)


class NoNorm(nn.Module):
    def __init__(self, feat_size):
        super(NoNorm, self).__init__()
        self.bias = nn.Parameter(torch.zeros(feat_size))
        self.weight = nn.Parameter(torch.ones(feat_size))

    def forward(self, input_tensor):
        return input_tensor * self.weight + self.bias


class MobileBertEmbedding(nn.Module):
    def __init__(self, config):
        super(MobileBertEmbedding, self).__init__()

        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.embed_size, padding_idx=config.pad_token_id
        )
        self.segment_embeddings = nn.Embedding(
            config.segment_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.position_size, config.hidden_size
        )
        self.dense = nn.Linear(config.embed_size * 3, config.hidden_size)
        self.layernorm = (
            nn.LayerNorm(config.hidden_size)
            if not config.use_opt
            else NoNorm(config.hidden_size)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, token_ids, segment_ids, position_ids):
        token_embeddings = self.token_embeddings(token_ids)
        token_embeddings = torch.cat(
            [
                F.pad(token_embeddings[:, 1:], [0, 0, 0, 1, 0, 0], value=0),
                token_embeddings,
                F.pad(token_embeddings[:, :-1], [0, 0, 1, 0, 0, 0], value=0),
            ],
            dim=2,
        )
        token_embeddings = self.dense(token_embeddings)

        segment_embeddings = self.segment_embeddings(segment_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = token_embeddings + segment_embeddings + position_embeddings
        embeddings = self.dropout(self.layernorm(embeddings))
        return embeddings


class MobileBertAttention(nn.Module):
    def __init__(self, config):
        super(MobileBertAttention, self).__init__()

        self.num_attn_heads = config.num_attn_heads
        # logging.info("MobileBertAttention, j={}, self.num_attn_heads={}".format(j, self.num_attn_heads))
        self.attn_head_size = config.inner_hidden_size // self.num_attn_heads
        self.all_head_size = self.attn_head_size * self.num_attn_heads

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(config.attn_dropout_prob)

        self.softmax = nn.Softmax(dim=-1)
        self.dense = nn.Linear(self.all_head_size, config.inner_hidden_size)
        self.layernorm = (
            nn.LayerNorm(config.inner_hidden_size)
            if not config.use_opt
            else NoNorm(config.inner_hidden_size)
        )
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states,
        inner_hidden_states,
        attn_mask,
        is_calc_cka=False,
        get_cka_pair=False,
    ):
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        query = query.view(
            hidden_states.size(0), -1, self.num_attn_heads, self.attn_head_size
        ).permute(0, 2, 1, 3)
        key = key.view(
            hidden_states.size(0), -1, self.num_attn_heads, self.attn_head_size
        ).permute(0, 2, 1, 3)
        value = value.view(
            hidden_states.size(0), -1, self.num_attn_heads, self.attn_head_size
        ).permute(0, 2, 1, 3)

        # print("attn_mask.shape={}".format(attn_mask.shape))
        # print("attn_mask={}".format(attn_mask))
        # for i in range(attn_mask.size(0)):
        #    for j in range(attn_mask.size(0)):
        #       print(attn_mask[i][j])
        attn_mask = attn_mask[:, None, None, :]
        # print("attn_mask.shape={}".format(attn_mask.shape))
        # print("attn_mask={}".format(attn_mask))
        attn_score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
            self.attn_head_size
        )
        attn_score += attn_mask * -10000.0
        attn_prob = self.attn_dropout(self.softmax(attn_score))

        context = torch.matmul(attn_prob, value)
        # [64, 12, 128, 44]
        # print(context.size())
        c = (
            context.permute(1, 0, 2, 3)
            .contiguous()
            .view(self.num_attn_heads, value.size(0), -1)
        )
        # print(c.size())
        # print(len(c))
        context = (
            context.permute(0, 2, 1, 3)
            .contiguous()
            .view(value.size(0), -1, self.all_head_size)
        )
        # print(context.shape)
        avg_cka = 0
        if get_cka_pair:
            is_calc_cka = True
        if is_calc_cka:
            # for now_batch in range(context.shape[0]):
            #        for i in range(self.num_attn_heads):
            #            for j in range(self.num_attn_heads):
            #                cka_sum[i][j] += cuda_cka.kernel_CKA(context[now_batch,:,i*self.attn_head_size:(i+1)*self.attn_head_size],
            #                                                     context[now_batch,:,j*self.attn_head_size:(j+1)*self.attn_head_size])
            with torch.no_grad():
                cka_logger = CKA_Minibatch_Grid(
                    self.num_attn_heads, self.num_attn_heads
                )
                # for data_batch in data_loader:
                #     feature_list_1 = model_1(data_batch)  # len(feature_list_1) = d1
                #     feature_list_2 = model_2(data_batch)  # len(feature_list_2) = d2
                cka_logger.update(c, c)
                cka_sum = cka_logger.compute()  # [d1, d2]
                # logging.info("cka_sum={}".format(cka_sum))
                numpy_cka = cka_sum.sum().numpy()
                avg_cka = (numpy_cka - self.num_attn_heads) / (
                    self.num_attn_heads * self.num_attn_heads - self.num_attn_heads
                )
            # print('cka_sum_MobileBertAttention={}'.format(cka_sum))
        context = self.hidden_dropout(self.dense(context))
        output = self.layernorm(inner_hidden_states + context)
        if get_cka_pair:
            return output, attn_score, avg_cka, cka_sum
        return output, attn_score, avg_cka

    def flops(self, seq_len):
        flops = 0
        flops += 3 * seq_len * self.query.in_features * self.query.out_features
        flops += seq_len * seq_len * self.all_head_size
        flops += seq_len * seq_len * self.all_head_size
        flops += seq_len * self.dense.in_features * self.dense.out_features
        return flops


class TinyMobileBertAttention(nn.Module):
    def __init__(self, config):
        super(TinyMobileBertAttention, self).__init__()

        self.num_attn_heads = config.num_attn_heads
        self.attn_head_size = config.inner_hidden_size // self.num_attn_heads
        self.all_head_size = self.attn_head_size * self.num_attn_heads

        self.query = nn.Linear(config.inner_hidden_size, self.all_head_size)
        # self.key = nn.Linear(config.inner_hidden_size, self.all_head_size)
        self.key = self.query
        self.value = nn.Linear(config.inner_hidden_size, self.all_head_size)
        self.attn_dropout = nn.Dropout(config.attn_dropout_prob)

        self.softmax = nn.Softmax(dim=-1)
        self.dense = nn.Linear(self.all_head_size, config.inner_hidden_size)
        self.layernorm = (
            nn.LayerNorm(config.inner_hidden_size)
            if not config.use_opt
            else NoNorm(config.inner_hidden_size)
        )
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attn_mask):
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        query = query.view(
            hidden_states.size(0), -1, self.num_attn_heads, self.attn_head_size
        ).permute(0, 2, 1, 3)
        key = key.view(
            hidden_states.size(0), -1, self.num_attn_heads, self.attn_head_size
        ).permute(0, 2, 1, 3)
        value = value.view(
            hidden_states.size(0), -1, self.num_attn_heads, self.attn_head_size
        ).permute(0, 2, 1, 3)

        attn_mask = attn_mask[:, None, None, :]
        attn_score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
            self.attn_head_size
        )
        attn_score += attn_mask * -10000.0
        attn_prob = self.attn_dropout(self.softmax(attn_score))

        context = torch.matmul(attn_prob, value)
        context = (
            context.permute(0, 2, 1, 3)
            .contiguous()
            .view(value.size(0), -1, self.all_head_size)
        )
        context = self.hidden_dropout(self.dense(context))
        output = self.layernorm(hidden_states + context)
        return output, attn_score


class MobileBertFeedForwardNetwork(nn.Module):
    def __init__(self, config, j):
        super(MobileBertFeedForwardNetwork, self).__init__()

        self.max_ffn_hidden_size = config.max_ffn_hidden_size
        self.init_hidden_size = config.init_hidden_size
        self.hidden_size_increment = config.hidden_size_increment
        self.dense1 = DynamicLinear(
            config.inner_hidden_size, self.max_ffn_hidden_size)
        self.activation = models.gelu if not config.use_opt else nn.ReLU()
        self.dense2 = DynamicLinear(
            self.max_ffn_hidden_size, config.inner_hidden_size)
        self.dense1.set_indices(
            [0, config.inner_hidden_size - 1], [0,
                                                self.init_hidden_size * (j + 1) - 1]
        )
        self.dense2.set_indices(
            [0, self.init_hidden_size *
                (j + 1) - 1], [0, config.inner_hidden_size - 1]
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layernorm = (
            nn.LayerNorm(config.inner_hidden_size)
            if not config.use_opt
            else NoNorm(config.inner_hidden_size)
        )

    def add_indices(self):
        logging.info(
            'MobileBertFeedForwardNetwork, self.dense1.get_out_dimension()={}, self.max_ffn_hidden_size={}'.format(
                self.dense1.get_out_dimension(), self.max_ffn_hidden_size
            )
        )
        if self.dense1.get_out_dimension() >= self.max_ffn_hidden_size:
            return False
        self.dense1.add_out_indices(self.hidden_size_increment)
        self.dense2.add_in_indices(self.hidden_size_increment)
        return True

    def forward(self, hidden_states, is_calc_pca=True, min_pca=1):
        # output = self.activation(self.dense1(hidden_states))
        output = self.dense1(hidden_states)
        # logging.info('MobileBertTransformerBlockForSupernet out_size={}'.format(out_size))
        pca_sum = 0
        in_size = self.dense1.get_in_dimension().item()
        out_size = self.dense1.get_out_dimension().item()
        if is_calc_pca:
            pca_sum = pca_torch(output.view(-1, out_size), (0.99,))[0]
        output = self.activation(output)
        output = self.dropout(self.dense2(output))
        output = self.layernorm(hidden_states + output)
        if min_pca == 1:
            return output, pca_sum / min(in_size, out_size)
        else:
            return output, pca_sum / out_size

    def flops(self, seq_len):
        flops = 0
        flops += (
            self.dense1.get_in_dimension() * self.dense1.get_out_dimension() * seq_len
        )
        flops += (
            self.dense2.get_in_dimension() * self.dense2.get_out_dimension() * seq_len
        )
        return flops


class MobileBertTransformerBlockForSupernet(nn.Module):
    def __init__(self, config, j):
        super(MobileBertTransformerBlockForSupernet, self).__init__()

        self.dense1 = nn.Linear(config.hidden_size, config.inner_hidden_size)
        self.dense2 = nn.Linear(config.inner_hidden_size, config.hidden_size)
        self.attention = MobileBertAttention(config)
        self.type = j
        self.ffn = MobileBertFeedForwardNetwork(config, j)
        self.layernorm = (
            nn.LayerNorm(config.hidden_size)
            if not config.use_opt
            else NoNorm(config.hidden_size)
        )

    def add_indices(self):
        return self.ffn.add_indices()

    def forward(
        self,
        hidden_states,
        attn_mask,
        is_calc_pca_cka=False,
        min_pca=1,
        get_cka_pair=False,
    ):
        # logging.info('config.hidden_size={}, config.inner_hidden_size={}, hidden_states.shape={}'.format(
        #    self.dense1.in_features, self.dense1.out_features, hidden_states.shape))
        inner_hidden_states = self.dense1(hidden_states)
        if get_cka_pair:
            attn_output, attn_score, cka_sum, cka_pair = self.attention(
                hidden_states,
                inner_hidden_states,
                attn_mask,
                is_calc_pca_cka,
                get_cka_pair=get_cka_pair,
            )
        else:
            attn_output, attn_score, cka_sum = self.attention(
                hidden_states, inner_hidden_states, attn_mask, is_calc_pca_cka
            )
        output = attn_output
        output, pca_sum = self.ffn(output, is_calc_pca_cka, min_pca)
        output = self.layernorm(hidden_states + self.dense2(output))
        # import logging
        # logging.info('output.shape={}, attn_score.shape={}'.format(output.shape, attn_score.shape))
        if get_cka_pair:
            return output, attn_output, attn_score, cka_sum, pca_sum, cka_pair
        # logging.info("Mobile pca_sum={}".format(pca_sum))
        return output, attn_output, attn_score, cka_sum, pca_sum

    def params(self):
        p = 0
        # logging.info("MobileBertTransformerBlockForSupernet self.type={} modules()={}".format(self.type, self.modules()))
        for m in self.modules():
            if isinstance(m, DynamicLinear):
                in_features = m.in_indices[1] - m.in_indices[0] + 1
                out_features = m.out_indices[1] - m.out_indices[0] + 1
                p += in_features * out_features + out_features
            elif isinstance(m, nn.Linear):
                p += m.in_features * m.out_features + m.out_features
            elif isinstance(m, nn.LayerNorm):
                p += m.normalized_shape[0] + m.normalized_shape[0]
        # if self.type // 3 != 0:
        #    p -= self.attention.key.in_features * self.attention.key.out_features + self.attention.key.out_features
        return p

    def flops(self, seq_len):
        flops = 0
        flops += seq_len * self.dense1.in_features * self.dense1.out_features
        flops += self.attention.flops(seq_len)
        flops += self.ffn.flops(seq_len)
        flops += seq_len * self.dense2.in_features * self.dense2.out_features
        return flops


class TinyMobileBertTransformerBlock(nn.Module):
    def __init__(self, config):
        super(TinyMobileBertTransformerBlock, self).__init__()

        self.dense1 = nn.Linear(config.hidden_size, config.inner_hidden_size)
        self.dense2 = nn.Linear(config.inner_hidden_size, config.hidden_size)
        self.attention = TinyMobileBertAttention(config)
        self.all_ffn = nn.ModuleList(
            [MobileBertFeedForwardNetwork(config) for _ in range(2)]
        )
        self.layernorm = (
            nn.LayerNorm(config.hidden_size)
            if not config.use_opt
            else NoNorm(config.hidden_size)
        )

    def forward(self, hidden_states, attn_mask):
        inner_hidden_states = self.dense1(hidden_states)
        attn_output, attn_score = self.attention(
            inner_hidden_states, attn_mask)
        output = attn_output
        for ffn_layer in self.all_ffn:
            output = ffn_layer(output)
        output = self.layernorm(hidden_states + self.dense2(output))
        return output, attn_score


class MobileBertSingle(nn.Module):
    def __init__(self, config, use_lm=False):
        super(MobileBertSingle, self).__init__()

        self.use_lm = use_lm
        self.embeddings = MobileBertEmbedding(config)
        if config.is_tiny:
            self.encoder = nn.ModuleList(
                [
                    TinyMobileBertTransformerBlock(config)
                    for _ in range(config.num_layers)
                ]
            )
        else:
            self.encoder = nn.ModuleList(
                [MobileBertTransformerBlock(config)
                 for _ in range(config.num_layers)]
            )

        if self.use_lm:
            self.lm_head = models.BertMaskedLMHead(
                config, self.embeddings.token_embeddings.weight
            )
        self._init_weights()

    def forward(self, token_ids, segment_ids, position_ids, attn_mask):
        all_attn_outputs, all_ffn_outputs = [], []
        output = self.embeddings(token_ids, segment_ids, position_ids)
        all_ffn_outputs.append(output)

        for layer in self.encoder:
            output, attn_output = layer(output, attn_mask)
            all_attn_outputs.append(attn_output)
            all_ffn_outputs.append(output)

        if self.use_lm:
            output = self.lm_head(output)
        return output, all_attn_outputs, all_ffn_outputs

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()


class MobileBert(nn.Module):
    def __init__(self, config, task, return_hid=False):
        super(MobileBert, self).__init__()

        self.task = task
        self.return_hid = return_hid
        self.embeddings = MobileBertEmbedding(config)
        if config.is_tiny:
            self.encoder = nn.ModuleList(
                [
                    TinyMobileBertTransformerBlock(config)
                    for _ in range(config.num_layers)
                ]
            )
        else:
            self.encoder = nn.ModuleList(
                [MobileBertTransformerBlock(config)
                 for _ in range(config.num_layers)]
            )

        if task in datasets.glue_tasks:
            self.num_classes = datasets.glue_num_classes[task]
            self.cls_pooler = models.BertClsPooler(config)
        elif task in datasets.squad_tasks:
            self.num_classes = 2
        elif task in datasets.multi_choice_tasks:
            self.num_classes = 1
            self.cls_pooler = models.BertClsPooler(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_classes)
        self._init_weights()

    def forward(self, token_ids, segment_ids, position_ids, attn_mask):
        if self.task in datasets.multi_choice_tasks:
            num_choices = token_ids.size(1)
            token_ids = token_ids.view(-1, token_ids.size(-1))
            segment_ids = segment_ids.view(-1, segment_ids.size(-1))
            position_ids = position_ids.view(-1, position_ids.size(-1))
            attn_mask = attn_mask.view(-1, attn_mask.size(-1))

        all_attn_outputs, all_ffn_outputs = [], []
        output = self.embeddings(token_ids, segment_ids, position_ids)
        all_ffn_outputs.append(output)

        for layer in self.encoder:
            output, attn_output = layer(output, attn_mask)
            all_attn_outputs.append(attn_output)
            all_ffn_outputs.append(output)

        if self.task in datasets.glue_tasks:
            output = self.cls_pooler(output[:, 0])
            output = self.classifier(output).squeeze(-1)
            if self.return_hid:
                return output, all_attn_outputs, all_ffn_outputs
            return output
        elif self.task in datasets.squad_tasks:
            output = self.classifier(output)
            start_logits, end_logits = output.split(1, dim=-1)
            if self.return_hid:
                return (
                    start_logits.squeeze(-1),
                    end_logits.squeeze(-1),
                    all_attn_outputs,
                    all_ffn_outputs,
                )
            return start_logits.squeeze(-1), end_logits.squeeze(-1)
        elif self.task in datasets.multi_choice_tasks:
            output = self.cls_pooler(output[:, 0])
            output = self.classifier(output)
            output = output.view(-1, num_choices)
            if self.return_hid:
                return output, all_attn_outputs, all_ffn_outputs
            return output

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
