import torch.nn as nn
import datasets
import models
from .mobile_bert import MobileBertTransformerBlockForSupernet
from .transformer_multibranch_v2 import TransformerEncoderLayer as MultiBranchBlockForSupernet
from utils import calc_params
from sklearn.decomposition import PCA
import numpy as np
from .pca_torch import pca_torch



class MySupernetFeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super(MySupernetFeedForwardNetwork, self).__init__()

        ffn_hidden_size = int(config.ffn_hidden_size * config.ffn_expansion_ratio)
        self.dense1 = nn.Linear(config.hidden_size, ffn_hidden_size)
        self.activation = models.gelu
        self.dense2 = nn.Linear(ffn_hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states, is_calc_pca):
        output = self.activation(self.dense1(hidden_states))
        pca_sum = np.zeros(min(output.shape[1], output.shape[2]));
        if is_calc_pca:
            for i in range(output.shape[0]):
                pca_line = PCA().fit(output[i].clone().detach().cpu().numpy())
                pca_sum += np.cumsum(pca_line.explained_variance_ratio_)
            pca_sum /= output.shape[0]
        output = self.dropout(self.dense2(output))
        output = self.layernorm(hidden_states + output)
        return output, pca_sum

    def flops(self, seq_len):
        flops = 0
        flops += self.dense1.in_features * self.dense1.out_features * seq_len
        flops += self.dense2.in_features * self.dense2.out_features * seq_len
        return flops


class MySupernetTransformerBlock(nn.Module):
    def __init__(self, config):
        super(MySupernetTransformerBlock, self).__init__()

        self.attention = models.MySupernetBertAttention(config)
        self.ffn = MySupernetFeedForwardNetwork(config)

    def forward(self, hidden_states, attn_mask, is_calc_pca_cka):
        output, attn_score, cka_sum = self.attention(hidden_states, attn_mask, is_calc_pca_cka)
        output, pca_sum = self.ffn(output, is_calc_pca_cka)
        return output, attn_score, cka_sum, pca_sum

    def flops(self, seq_len):
        flops = 0
        flops += self.attention.flops(seq_len)
        flops += self.ffn.flops(seq_len)
        return flops


class MySupernetSingle(nn.Module):
    def __init__(self, config, use_lm=False, ret_all_ffn_hidden_states=False):
        super(MySupernetSingle, self).__init__()

        self.use_fit_dense = config.use_fit_dense
        self.ret_all_ffn_hidden_states = ret_all_ffn_hidden_states
        self.use_lm = use_lm
        if config.use_mobile_embed:
            self.embeddings = models.MobileBertEmbedding(config)
        else:
            self.embeddings = models.BertEmbedding(config)


        config_mobile_bert = models.select_config('mobile_bert_for_supernet', True)
        config_multi_branch = models.select_config('multi_branch', True)
        self.num_layers = config.num_layers
        layers = []
        for i in range(config.num_layers):
            layer = nn.ModuleList([MySupernetTransformerBlock(config), MobileBertTransformerBlockForSupernet(config_mobile_bert),
                                   MultiBranchBlockForSupernet(config_multi_branch)])
            layers.append(layer)

        self.encoder = nn.Sequential(*layers)

        if self.use_fit_dense:
            self.fit_dense = nn.Linear(config.hidden_size, config.fit_size)
        if self.use_lm:
            self.lm_head = models.BertMaskedLMHead(config, self.embeddings.token_embeddings.weight)
        self._init_weights()

    def forward(self, token_ids, segment_ids, position_ids, attn_mask, select_arch=[], is_calc_pca_cka = False):
        all_attn_outputs, all_ffn_outputs = [], []
        output = self.embeddings(token_ids, segment_ids, position_ids)
        if self.use_fit_dense:
            all_ffn_outputs.append(self.fit_dense(output))
        else:
            all_ffn_outputs.append(output)

        import random
        if select_arch == []:
            for i in range(self.num_layers):
                select_arch.append(random.randint(0, 1))

        import logging
        First = True
        for archs, arch_id in zip(self.encoder, select_arch):
            if First:
                output, attn_output, cka_sum, pca_sum = archs[arch_id](output, attn_mask, is_calc_pca_cka)
                First = False
            else:
                output, attn_output, one_cka, one_pca = archs[arch_id](output, attn_mask, is_calc_pca_cka)
                pca_sum += one_pca
                cka_sum += one_cka
           # logging.info('attn_output.shape={}'.format(attn_output.shape))
            all_attn_outputs.append(attn_output)
            if self.use_fit_dense:
                all_ffn_outputs.append(self.fit_dense(output))
            else:
                all_ffn_outputs.append(output)

        cka_sum /= self.num_layers
        pca_sum /= self.num_layers

        if self.use_lm:
            output = self.lm_head(output)

        return output, all_attn_outputs, all_ffn_outputs, cka_sum, pca_sum


    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()


class MySupernet(nn.Module):
    def __init__(self, config, task, return_hidden_states=False, ret_all_ffn_hidden_states=False):
        super(MySupernet, self).__init__()

        self.use_fit_dense = config.use_fit_dense
        self.ret_all_ffn_hidden_states = ret_all_ffn_hidden_states
        self.task = task
        self.return_hidden_states = return_hidden_states
        if config.use_mobile_embed:
            self.embeddings = models.MobileBertEmbedding(config)
        else:
            self.embeddings = models.BertEmbedding(config)

        config_mobile_bert = models.select_config('mobile_bert_for_supernet', True)
        config_multi_branch = models.select_config('multi_branch', True)
        self.num_layers = config.num_layers
        layers = []
        for i in range(config.num_layers):
            layer = nn.ModuleList([MySupernetTransformerBlock(config), MobileBertTransformerBlockForSupernet(config_mobile_bert),
                                   MultiBranchBlockForSupernet(config_multi_branch)])
            layers.append(layer)

        self.encoder = nn.Sequential(*layers)

        if self.use_fit_dense:
            self.fit_dense = nn.Linear(config.hidden_size, config.fit_size)
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

    def forward(self, token_ids, segment_ids, position_ids, attn_mask, select_arch=[]):
        if self.task in datasets.multi_choice_tasks:
            num_choices = token_ids.size(1)
            token_ids = token_ids.view(-1, token_ids.size(-1))
            segment_ids = segment_ids.view(-1, segment_ids.size(-1))
            position_ids = position_ids.view(-1, position_ids.size(-1))
            attn_mask = attn_mask.view(-1, attn_mask.size(-1))

        all_attn_outputs, all_ffn_outputs = [], []
        output = self.embeddings(token_ids, segment_ids, position_ids)
        if self.use_fit_dense:
            all_ffn_outputs.append(self.fit_dense(output))
        else:
            all_ffn_outputs.append(output)

        import random
        if select_arch == []:
            for i in range(self.num_layers):
                select_arch.append(random.randint(0, 1))

        import logging
        for archs, arch_id in zip(self.encoder, select_arch):
            output, attn_output = archs[arch_id](output, attn_mask)
           # logging.info('attn_output.shape={}'.format(attn_output.shape))
            all_attn_outputs.append(attn_output)
            if self.use_fit_dense:
                all_ffn_outputs.append(self.fit_dense(output))
            else:
                all_ffn_outputs.append(output)

        if self.task in datasets.glue_tasks:
            output = self.cls_pooler(output[:, 0])
            output = self.classifier(output).squeeze(-1)
            if self.return_hidden_states:
                return output, all_attn_outputs, all_ffn_outputs
            return output
        elif self.task in datasets.squad_tasks:
            output = self.classifier(output)
            start_logits, end_logits = output.split(1, dim=-1)
            if self.return_hidden_states:
                return start_logits.squeeze(-1), end_logits.squeeze(-1), all_attn_outputs, all_ffn_outputs
            return start_logits.squeeze(-1), end_logits.squeeze(-1)
        elif self.task in datasets.multi_choice_tasks:
            output = self.cls_pooler(output[:, 0])
            output = self.classifier(output)
            output = output.view(-1, num_choices)
            if self.return_hidden_states:
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


class MultiTaskMySupernet(nn.Module):
    def __init__(self, config, task, return_hidden_states=False, ret_all_ffn_hidden_states=False):
        super(MultiTaskMySupernet, self).__init__()

        self.use_fit_dense = config.use_fit_dense
        self.ret_all_ffn_hidden_states = ret_all_ffn_hidden_states
        self.task = task
        self.return_hidden_states = return_hidden_states
        if config.use_mobile_embed:
            self.embeddings = models.MobileBertEmbedding(config)
        else:
            self.embeddings = models.BertEmbedding(config)

        config_mobile_bert = models.select_config('mobile_bert_for_supernet', True)
        config_multi_branch = models.select_config('multi_branch', True)
        self.num_layers = config.num_layers
        layers = []
        for i in range(config.num_layers):
            layer = nn.ModuleList([MySupernetTransformerBlock(config), MobileBertTransformerBlockForSupernet(config_mobile_bert),
                                   MultiBranchBlockForSupernet(config_multi_branch)])
            layers.append(layer)

        self.encoder = nn.Sequential(*layers)

        if self.use_fit_dense:
            self.fit_dense = nn.Linear(config.hidden_size, config.fit_size)

        self.cls_pooler = models.BertClsPooler(config)
        self.classifiers = nn.ModuleList([])
        for task in datasets.glue_train_tasks:
            num_classes = datasets.glue_num_classes[task]
            self.classifiers.append(nn.Linear(config.hidden_size, num_classes))
        self._init_weights()

    def forward(self, task_id, token_ids, segment_ids, position_ids, attn_mask, select_arch=[]):
        if self.task in datasets.multi_choice_tasks:
            num_choices = token_ids.size(1)
            token_ids = token_ids.view(-1, token_ids.size(-1))
            segment_ids = segment_ids.view(-1, segment_ids.size(-1))
            position_ids = position_ids.view(-1, position_ids.size(-1))
            attn_mask = attn_mask.view(-1, attn_mask.size(-1))

        all_attn_outputs, all_ffn_outputs = [], []
        output = self.embeddings(token_ids, segment_ids, position_ids)
        if self.use_fit_dense:
            all_ffn_outputs.append(self.fit_dense(output))
        else:
            all_ffn_outputs.append(output)

        import random
        if select_arch == []:
            for i in range(self.num_layers):
                select_arch.append(random.randint(0, 1))

        import logging
        for archs, arch_id in zip(self.encoder, select_arch):
            output, attn_output = archs[arch_id](output, attn_mask)
           # logging.info('attn_output.shape={}'.format(attn_output.shape))
            all_attn_outputs.append(attn_output)
            if self.use_fit_dense:
                all_ffn_outputs.append(self.fit_dense(output))
            else:
                all_ffn_outputs.append(output)

        output = self.cls_pooler(output[:, 0])
        output = self.classifiers[task_id](output).squeeze(-1)
        if self.return_hidden_states:
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
            params += calc_params(block)
        params += calc_params(self.cls_pooler)
        for e in self.classifiers:
            params += e.in_features * e.out_features
        return params
