import logging

import datasets
import lpzero.model as models
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from lpzero.utils import calc_params

from .dynamic_ops import DynamicLinear
from .ls_model import TransformerLSLayer
from .mobile_bert import MobileBertTransformerBlockForSupernet
from .modeling_convbert import ConvBertLayer
from .pca_torch import pca_torch
# from .transformer_multibranch_v2 import (
#     TransformerEncoderLayer as MultiBranchBlockForSupernet,
# ) # TODO fairseq 


class MySupernetFeedForwardNetwork(nn.Module):
    def __init__(self, config, j):
        super(MySupernetFeedForwardNetwork, self).__init__()

        self.max_ffn_hidden_size = config.max_ffn_hidden_size
        self.init_hidden_size = config.init_hidden_size
        self.hidden_size_increment = config.hidden_size_increment
        self.dense1 = DynamicLinear(
            config.hidden_size, self.max_ffn_hidden_size)
        self.activation = models.gelu
        self.hidden_size = config.hidden_size
        # logging.info("config.hidden_size={}, config.ffn_expansion_ratio={}, self.mid_size={}".format(config.hidden_size, config.ffn_expansion_ratio, self.mid_size))
        self.dense2 = DynamicLinear(
            self.max_ffn_hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense1.set_indices(
            [0, config.hidden_size - 1], [0,
                                          self.init_hidden_size * (j + 1) - 1]
        )
        self.dense2.set_indices(
            [0, self.init_hidden_size *
                (j + 1) - 1], [0, config.hidden_size - 1]
        )

        self.layernorm = nn.LayerNorm(config.hidden_size)

    def add_indices(self):
        logging.info(
            'MySupernetFeedForwardNetwork, self.dense1.get_out_dimension()={}, self.max_ffn_hidden_size={}'.format(
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
        in_size = self.dense1.get_in_dimension().item()
        out_size = self.dense1.get_out_dimension().item()
        # logging.info('output.view(-1, out_size).shape={}'.format(output.view(-1, out_size).shape))
        # logging.info('MySupernetFeedForwardNetwork out_size={}'.format(out_size))
        pca_sum = 0
        if is_calc_pca:
            pca_sum = pca_torch(output.view(-1, out_size), (0.99,))[0]
            # print("pca_sum={}".format(pca_sum))
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


class MySupernetTransformerBlock(nn.Module):
    def __init__(self, config, j):
        super(MySupernetTransformerBlock, self).__init__()

        self.attention = models.MySupernetBertAttention(config)
        self.type = j
        self.ffn = MySupernetFeedForwardNetwork(config, j)

    def forward(
        self,
        hidden_states,
        attn_mask,
        is_calc_pca_cka=False,
        min_pca=1,
        get_cka_pair=False,
    ):
        if get_cka_pair:
            output, attn_score, one_cka, cka_pair = self.attention(
                hidden_states, attn_mask, is_calc_pca_cka, get_cka_pair=get_cka_pair
            )
        else:
            output, attn_score, one_cka = self.attention(
                hidden_states, attn_mask, is_calc_pca_cka
            )
        attn_output = output
        output, one_pca = self.ffn(output, is_calc_pca_cka, min_pca)
        if get_cka_pair:
            return output, attn_output, attn_score, one_cka, one_pca, cka_pair
        # logging.info("BERT one_cka={}".format(one_cka))
        return output, attn_output, attn_score, one_cka, one_pca

    def add_indices(self):
        return self.ffn.add_indices()

    def params(self):
        p = 0
        # logging.info(
        #    "MySupernetTransformerBlock self.type={} modules()={}".format(self.type, self.modules()))
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
        flops += self.attention.flops(seq_len)
        flops += self.ffn.flops(seq_len)
        return flops


class MySupernetSingle(nn.Module):
    def __init__(
        self,
        config,
        use_lm=False,
        ret_all_ffn_hidden_states=False,
        issingle=True,
        fixed_dimension=False,
    ):
        super(MySupernetSingle, self).__init__()

        self.use_fit_dense = config.use_fit_dense
        self.ret_all_ffn_hidden_states = ret_all_ffn_hidden_states
        self.use_lm = use_lm
        if config.use_mobile_embed:
            self.embeddings = models.MobileBertEmbedding(config)
        else:
            self.embeddings = models.BertEmbedding(config)

        # print("issingle={}".format(issingle))
        if config.name == 'MySupernetConfig_5M':
            config_mobile_bert = models.select_config(
                'mobile_bert_for_supernet_5M', True, issingle, fixed_dimension
            )
        else:
            config_mobile_bert = models.select_config(
                'mobile_bert_for_supernet', True, issingle, fixed_dimension
            )
        config_ls_transformer = models.select_config('lstransformer', True)
        config_convbert = models.select_config('convbert', True, issingle)
        # config_mobile_bert.print_configs()
        # config_convbert.print_configs()
        self.num_layers = config.num_layers
        self.type_each_block = config.type_each_block
        layers = []
        for i in range(config.num_layers):
            layer = nn.ModuleList([])
            for j in range(config.type_each_block):
                layer.append(MySupernetTransformerBlock(config, j))
            # print("config_mobile_bert.init_hidden_size={}".format(config_mobile_bert.init_hidden_size))
            for j in range(config.type_each_block):
                layer.append(
                    MobileBertTransformerBlockForSupernet(
                        config_mobile_bert, j)
                )
            layers.append(layer)

        self.encoder = nn.Sequential(*layers)

        if self.use_fit_dense:
            self.fit_dense = nn.Linear(config.hidden_size, config.fit_size)
            self.fit_dense_attn = nn.ModuleList(
                [
                    nn.Linear(config.hidden_size, config.fit_size),
                    nn.Linear(config_mobile_bert.inner_hidden_size,
                              config.fit_size),
                ]
            )
        if self.use_lm:
            self.lm_head = models.BertMaskedLMHead(
                config, self.embeddings.token_embeddings.weight
            )
        self._init_weights()

    def add_indices(self, i, j):
        return self.encoder[j][i].add_indices()

    def forward(
        self,
        token_ids,
        segment_ids,
        position_ids,
        attn_mask,
        type_blocks,
        select_arch=[],
        is_calc_pca_cka=False,
        min_pca=1,
        get_cka_pair=False,
    ):
        all_attn_outputs, all_attn_scores, all_ffn_outputs = [], [], []
        output = self.embeddings(token_ids, segment_ids, position_ids)
        initial_input = output
        if self.use_fit_dense:
            all_ffn_outputs.append(self.fit_dense(output))
        else:
            all_ffn_outputs.append(output)

        import random

        if select_arch == []:
            for i in range(self.num_layers):
                select_arch.append(random.randint(0, type_blocks))

        layer_idx = 0
        cka_each_layer = []
        for _ in range(type_blocks):
            cka_each_layer.append([0] * len(select_arch))
        pca_each_layer = []
        for _ in range(type_blocks):
            pca_each_layer.append([0] * len(select_arch))
        if get_cka_pair:
            cka_pairs = []
        for archs, arch_id in zip(self.encoder, select_arch):
            # if layer_idx == 0:
            # output, attn_output, cka_sum, pca_sum = archs[arch_id](output, attn_mask, is_calc_pca_cka)
            # logging.info("pca_sum={}, cka_sum={}".format(pca_sum,cka_sum))
            # else:
            # output, attn_output, one_cka, one_pca = archs[arch_id](output, attn_mask, is_calc_pca_cka)
            # logging.info("one_pca={}, one_cka={}".format(one_pca, one_cka))
            # pca_sum += one_pca
            # cka_sum += one_cka
            if get_cka_pair == False:
                output, attn_output, attn_score, one_cka, one_pca = archs[arch_id](
                    output, attn_mask, is_calc_pca_cka=is_calc_pca_cka, min_pca=min_pca
                )
            else:
                output, attn_output, attn_score, one_cka, one_pca, cka_pair = archs[
                    arch_id
                ](
                    output,
                    attn_mask,
                    is_calc_pca_cka=is_calc_pca_cka,
                    min_pca=min_pca,
                    get_cka_pair=get_cka_pair,
                )
                cka_pairs.append(cka_pair)
            cka_each_layer[arch_id][layer_idx] += one_cka
            pca_each_layer[arch_id][layer_idx] += one_pca
            # logging.info('attn_output.shape={}'.format(attn_output.shape))
            all_attn_scores.append(attn_score)
            if self.use_fit_dense:
                all_ffn_outputs.append(self.fit_dense(output))
                all_attn_outputs.append(
                    self.fit_dense_attn[arch_id //
                                        self.type_each_block](attn_output)
                )
            else:
                all_ffn_outputs.append(output)
                all_attn_outputs.append(attn_output)
            layer_idx += 1

        # cka_sum /= self.num_layers
        # pca_sum /= self.num_layers
        if get_cka_pair:
            layer_sim = []
            for i in range(self.num_layers):
                layer_output = []
                target_j = random.randint(0, type_blocks)
                for j in range(type_blocks):
                    output, _, _, _, _ = archs[j](
                        initial_input,
                        attn_mask,
                        is_calc_pca_cka=is_calc_pca_cka,
                        min_pca=min_pca,
                    )
                    if target_j == j:
                        initial_input = output
                    # logging.info("output.shape()={}".format(output.shape))
                    layer_output.append(output)
                sim_sum = 0
                import pandas as pd

                matrix = [
                    [None for _ in range(type_blocks)] for _ in range(type_blocks)
                ]
                for j in range(type_blocks):
                    for k in range(type_blocks):
                        if j != k:
                            import torch

                            one_sim = torch.mean(
                                F.cosine_similarity(
                                    layer_output[j], layer_output[k], dim=2
                                )
                            ).item()
                            sim_sum += one_sim
                            matrix[j][k] = one_sim
                            # print("j={}, k={}, one_sim={}", j, k, one_sim)
                if i == 0:
                    df = pd.DataFrame(matrix)
                    df.to_excel('output.xlsx', index=False)
                sim_sum /= type_blocks * type_blocks - type_blocks
                layer_sim.append(sim_sum)
            return layer_sim, cka_pairs

        if self.use_lm:
            output = self.lm_head(output)

        return (
            output,
            all_attn_scores,
            all_attn_outputs,
            all_ffn_outputs,
            cka_each_layer,
            pca_each_layer,
        )

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def params(self, select_arch):
        params = 0
        params += calc_params(self.embeddings)
        for i in range(len(self.encoder)):
            block = self.encoder[i][select_arch[i]]
            params += block.params()
        return params


class MySupernet(nn.Module):
    def __init__(
        self,
        config,
        task,
        return_hidden_states=False,
        ret_all_ffn_hidden_states=False,
        issingle=True,
    ):
        super(MySupernet, self).__init__()

        self.use_fit_dense = config.use_fit_dense
        self.ret_all_ffn_hidden_states = ret_all_ffn_hidden_states
        self.task = task
        self.return_hidden_states = return_hidden_states
        if config.use_mobile_embed:
            self.embeddings = models.MobileBertEmbedding(config)
        else:
            self.embeddings = models.BertEmbedding(config)

        if config.name == 'MySupernetConfig_5M':
            config_mobile_bert = models.select_config(
                'mobile_bert_for_supernet_5M', True, issingle
            )
        else:
            config_mobile_bert = models.select_config(
                'mobile_bert_for_supernet', True, issingle
            )
        config_convbert = models.select_config('convbert', True, issingle)
        # config_mobile_bert.print_configs()
        # config_convbert.print_configs()
        self.num_layers = config.num_layers
        layers = []
        self.type_each_block = config.type_each_block
        for i in range(config.num_layers):
            layer = nn.ModuleList([])
            for j in range(config.type_each_block):
                layer.append(MySupernetTransformerBlock(config, j))
            for j in range(config.type_each_block):
                layer.append(
                    MobileBertTransformerBlockForSupernet(
                        config_mobile_bert, j)
                )
            layers.append(layer)

        self.encoder = nn.Sequential(*layers)

        if self.use_fit_dense:
            self.fit_dense = nn.Linear(config.hidden_size, config.fit_size)
            self.fit_dense_attn = nn.ModuleList(
                [
                    nn.Linear(config.hidden_size, config.fit_size),
                    nn.Linear(config_mobile_bert.inner_hidden_size,
                              config.fit_size),
                ]
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

    def forward(
        self,
        token_ids,
        segment_ids,
        position_ids,
        attn_mask,
        type_blocks,
        select_arch=[],
        is_calc_pca_cka=False,
    ):
        if self.task in datasets.multi_choice_tasks:
            num_choices = token_ids.size(1)
            token_ids = token_ids.view(-1, token_ids.size(-1))
            segment_ids = segment_ids.view(-1, segment_ids.size(-1))
            position_ids = position_ids.view(-1, position_ids.size(-1))
            attn_mask = attn_mask.view(-1, attn_mask.size(-1))

        all_attn_outputs, all_ffn_outputs, all_attn_scores = [], [], []
        output = self.embeddings(token_ids, segment_ids, position_ids)
        if self.use_fit_dense:
            all_ffn_outputs.append(self.fit_dense(output))
        else:
            all_ffn_outputs.append(output)

        import random

        if select_arch == []:
            for i in range(self.num_layers):
                select_arch.append(random.randint(0, type_blocks))

        layer_idx = 0
        cka_each_layer = []
        for _ in range(type_blocks):
            cka_each_layer.append([0] * len(select_arch))
        pca_each_layer = []
        for _ in range(type_blocks):
            pca_each_layer.append([0] * len(select_arch))
        import logging

        for archs, arch_id in zip(self.encoder, select_arch):
            output, attn_output, attn_score, one_cka, one_pca = archs[arch_id](
                output, attn_mask, is_calc_pca_cka=is_calc_pca_cka
            )
            cka_each_layer[arch_id][layer_idx] += one_cka
            pca_each_layer[arch_id][layer_idx] += one_pca
            # logging.info('attn_output.shape={}'.format(attn_output.shape))
            all_attn_scores.append(attn_score)
            if self.use_fit_dense:
                all_ffn_outputs.append(self.fit_dense(output))
                all_attn_outputs.append(
                    self.fit_dense_attn[arch_id //
                                        self.type_each_block](attn_output)
                )
                # logging.info('attn_output.shape={}'.format(attn_output.shape))
                # logging.info('self.fit_dense_attn[arch_id](attn_output).shape={}'.format(self.fit_dense_attn[arch_id](attn_output).shape))
            else:
                all_ffn_outputs.append(output)
                all_attn_outputs.append(attn_output)
            layer_idx += 1

        if self.task in datasets.glue_tasks:
            output = self.cls_pooler(output[:, 0])
            output = self.classifier(output).squeeze(-1)
            if self.return_hidden_states:
                return (
                    output,
                    all_attn_outputs,
                    all_ffn_outputs,
                    cka_each_layer,
                    pca_each_layer,
                )
            return output, cka_each_layer, pca_each_layer
        elif self.task in datasets.squad_tasks:
            output = self.classifier(output)
            start_logits, end_logits = output.split(1, dim=-1)
            if self.return_hidden_states:
                return (
                    start_logits.squeeze(-1),
                    end_logits.squeeze(-1),
                    all_attn_outputs,
                    all_ffn_outputs,
                    cka_each_layer,
                    pca_each_layer,
                )
            return (
                start_logits.squeeze(-1),
                end_logits.squeeze(-1),
                cka_each_layer,
                pca_each_layer,
            )
        elif self.task in datasets.multi_choice_tasks:
            output = self.cls_pooler(output[:, 0])
            output = self.classifier(output)
            output = output.view(-1, num_choices)
            if self.return_hidden_states:
                return (
                    output,
                    all_attn_outputs,
                    all_ffn_outputs,
                    cka_each_layer,
                    pca_each_layer,
                )
            return output, cka_each_layer, pca_each_layer

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()


class MultiTaskMySupernet(nn.Module):
    def __init__(
        self,
        config,
        task,
        return_hidden_states=False,
        ret_all_ffn_hidden_states=False,
        issingle=False,
    ):
        super(MultiTaskMySupernet, self).__init__()

        self.use_fit_dense = config.use_fit_dense
        self.ret_all_ffn_hidden_states = ret_all_ffn_hidden_states
        self.task = task
        self.return_hidden_states = return_hidden_states
        if config.use_mobile_embed:
            self.embeddings = models.MobileBertEmbedding(config)
        else:
            self.embeddings = models.BertEmbedding(config)

        if config.name == 'MySupernetConfig_5M':
            config_mobile_bert = models.select_config(
                'mobile_bert_for_supernet_5M', True, issingle
            )
        else:
            config_mobile_bert = models.select_config(
                'mobile_bert_for_supernet', True, issingle
            )
        config_ls_transformer = models.select_config('lstransformer', True)
        config_convbert = models.select_config('convbert', True, issingle)
        self.num_layers = config.num_layers
        layers = []
        self.type_each_block = config.type_each_block
        for i in range(config.num_layers):
            layer = nn.ModuleList([])
            for j in range(config.type_each_block):
                layer.append(MySupernetTransformerBlock(config, j))
            for j in range(config.type_each_block):
                layer.append(
                    MobileBertTransformerBlockForSupernet(
                        config_mobile_bert, j)
                )
            layers.append(layer)

        self.encoder = nn.Sequential(*layers)

        if self.use_fit_dense:
            self.fit_dense = nn.Linear(config.hidden_size, config.fit_size)
            self.fit_dense_attn = nn.ModuleList(
                [
                    nn.Linear(config.hidden_size, config.fit_size),
                    nn.Linear(config_mobile_bert.inner_hidden_size,
                              config.fit_size),
                ]
            )

        self.cls_pooler = models.BertClsPooler(config)
        self.classifiers = nn.ModuleList([])
        for task in datasets.glue_train_tasks:
            num_classes = datasets.glue_num_classes[task]
            self.classifiers.append(nn.Linear(config.hidden_size, num_classes))
        self._init_weights()

    def add_indices(self, i, j):
        return self.encoder[j][i].add_indices()

    def forward(
        self,
        task_id,
        token_ids,
        segment_ids,
        position_ids,
        attn_mask,
        type_blocks,
        select_arch=[],
        is_spos=False,
    ):
        if self.task in datasets.multi_choice_tasks:
            num_choices = token_ids.size(1)
            token_ids = token_ids.view(-1, token_ids.size(-1))
            segment_ids = segment_ids.view(-1, segment_ids.size(-1))
            position_ids = position_ids.view(-1, position_ids.size(-1))
            attn_mask = attn_mask.view(-1, attn_mask.size(-1))

        all_attn_outputs, all_attn_scores, all_ffn_outputs = [], [], []
        output = self.embeddings(token_ids, segment_ids, position_ids)
        if self.use_fit_dense:
            all_ffn_outputs.append(self.fit_dense(output))
        else:
            all_ffn_outputs.append(output)

        import random

        if select_arch == []:
            for i in range(self.num_layers):
                select_arch.append(random.randint(0, type_blocks))

        import logging

        # logging.info('select_arch={}'.format(select_arch))
        layer_idx = 0
        sum_pca = 0
        # logging.info("select_arch={}".format(select_arch))
        for archs, arch_id in zip(self.encoder, select_arch):
            # logging.info('attn_output.shape={}'.format(attn_output.shape))
            output, attn_output, attn_score, one_cka, one_pca = archs[arch_id](
                output, attn_mask, is_calc_pca_cka=is_spos
            )
            sum_pca += one_pca
            # print("layer_idx={}, one_pca={}, sum_pca={}".format(layer_idx, one_pca, sum_pca))
            all_attn_scores.append(attn_score)
            if self.use_fit_dense:
                all_ffn_outputs.append(self.fit_dense(output))
                all_attn_outputs.append(
                    self.fit_dense_attn[arch_id //
                                        self.type_each_block](attn_output)
                )
                # logging.info("arch_id={},output={}, attn_output={}, all_ffn_outputs[-1]={}, all_attn_outputs[-1]={}".
                #            format(arch_id, output, attn_output, all_ffn_outputs[-1], all_attn_outputs[-1]))
                # logging.info('attn_output.shape={}'.format(attn_output.shape))
                # logging.info('self.fit_dense_attn[arch_id](attn_output).shape={}'.format(self.fit_dense_attn[arch_id](attn_output).shape))
            else:
                all_ffn_outputs.append(output)
                all_attn_outputs.append(attn_output)
            layer_idx += 1

        output = self.cls_pooler(output[:, 0])
        output = self.classifiers[task_id](output).squeeze(-1)
        if self.return_hidden_states:
            return output, all_attn_outputs, all_ffn_outputs
        # logging.info("sum_pca={}".format(sum_pca))
        return output, sum_pca

    def get_out_size_list(self, select_arch):
        out_size_list = []
        for archs, arch_id in zip(self.encoder, select_arch):
            out_size_list.append(
                archs[arch_id].ffn.dense1.get_out_dimension().item())
        return out_size_list

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def flops(self, select_arch, seq_len):
        flops = 0
        for i in range(len(self.encoder)):
            block = self.encoder[i][select_arch[i]]
            flops += block.flops(seq_len)
        return flops

    def params(self, select_arch):
        params = 0
        params += calc_params(self.embeddings)
        for i in range(len(self.encoder)):
            block = self.encoder[i][select_arch[i]]
            params += block.params()
        params += calc_params(self.cls_pooler)
        for e in self.classifiers:
            params += e.in_features * e.out_features
        return params
