"""
@Time   :   2020-11-25 14:27:59
@File   :   modeling.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
"""PyTorch ConvBert model. """

# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->Electra




import logging
import math
import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from .cka import CKA_Minibatch_Grid
from .dynamic_ops import DynamicLinear
from .pca_torch import pca_torch
class ConvBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, 'embedding_size'
        ):
            raise ValueError(
                'The hidden size (%d) is not a multiple of the number of attention '
                'heads (%d)' % (config.hidden_size, config.num_attention_heads)
            )

        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        new_num_attention_heads = int(
            config.num_attention_heads / config.head_ratio)
        if new_num_attention_heads < 1:
            config.head_ratio = config.num_attention_heads
            self.num_attention_heads = 1
        else:
            self.num_attention_heads = new_num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.key = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        mixed_query_layer,
        mixed_value_layer,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        is_calc_cka=False,
    ):
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)
        # logging.info('attention_scores.shape={}'.format(attention_scores.shape))
        attention_mask = attention_mask[:, None, None, :]
        # logging.info('attention_mask.shape={}'.format(attention_mask.shape))
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in ElectraModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        c = (
            context_layer.permute(1, 0, 2, 3)
            .contiguous()
            .view(self.num_attention_heads, value_layer.size(0), -1)
        )

        avg_cka = 0
        if is_calc_cka:
            # for now_batch in range(context.shape[0]):
            #        for i in range(self.num_attn_heads):
            #            for j in range(self.num_attn_heads):
            #                cka_sum[i][j] += cuda_cka.kernel_CKA(context[now_batch,:,i*self.attn_head_size:(i+1)*self.attn_head_size],
            #                                                     context[now_batch,:,j*self.attn_head_size:(j+1)*self.attn_head_size])
            with torch.no_grad():
                cka_logger = CKA_Minibatch_Grid(
                    self.num_attention_heads, self.num_attention_heads
                )
                # for data_batch in data_loader:
                #     feature_list_1 = model_1(data_batch)  # len(feature_list_1) = d1
                #     feature_list_2 = model_2(data_batch)  # len(feature_list_2) = d2
                cka_logger.update(c, c)
                cka_sum = cka_logger.compute()  # [d1, d2]
                numpy_cka = cka_sum.sum().numpy()
                avg_cka = (numpy_cka - self.num_attention_heads) / (
                    self.num_attention_heads * self.num_attention_heads
                    - self.num_attention_heads
                )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return context_layer, attention_scores, avg_cka


class SeparableConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, bias=True):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = kernel_size

        self.depthwise = nn.Conv1d(
            in_channels=self.in_ch,
            out_channels=self.in_ch,
            kernel_size=self.k,
            groups=self.in_ch,
            padding=self.k // 2,
            bias=False,
        )
        self.pointwise = nn.Conv1d(
            in_channels=self.in_ch,
            out_channels=self.out_ch,
            kernel_size=1,
            padding=0,
            bias=False,
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_ch))
        else:
            self.register_parameter('bias', None)
        nn.init.kaiming_normal_(self.depthwise.weight)
        nn.init.kaiming_normal_(self.pointwise.weight)

    def forward(self, x):
        out = self.pointwise(self.depthwise(x.transpose(1, 2))).transpose(1, 2)
        if self.bias is not None:
            out += self.bias
        return out


class ConvBertLightConv(nn.Module):
    def __init__(self, config, attention_head_size):
        super().__init__()
        self.attention_head_size = attention_head_size
        self.conv_kernel_size = config.conv_kernel_size
        self.unfold1d = nn.Unfold(
            kernel_size=[config.conv_kernel_size, 1],
            padding=[config.conv_kernel_size // 2, 0],
        )

    def forward(self, inputs, filters):
        # inputs(bs,seq_len, all_attention_heads),filters(bs, seq_len, num_heads*ks )
        bs, seqlen = inputs.shape[:2]
        conv_kernel_layer = filters.reshape(
            -1, self.conv_kernel_size, 1
        )  # [B * N * num_heads, ks, 1]
        conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        # conv_out_layer
        unfold_conv_out_layer = self.unfold1d(
            inputs.transpose(1, 2).contiguous().unsqueeze(-1)
        )  # [B, N, C] -> [B, C, N] -> [B, C, N * ks]
        # [1, 2, 3, 4, 5] -> unfold(ks=3, p=1) ->
        # [0, 1, 2, 3, 4, 5, 0] -> [0, 1, 2] [1, 2, 3] [2, 3, 4] -> [5 * 3]
        unfold_conv_out_layer = unfold_conv_out_layer.transpose(1, 2).reshape(
            bs, seqlen, -1, self.conv_kernel_size
        )  # [B, N * ks, C] -> [B, N, C, ks]
        conv_out_layer = torch.reshape(
            unfold_conv_out_layer, [-1,
                                    self.attention_head_size, self.conv_kernel_size]
        )  # [B * N * num_heads, C // num_heads, ks]
        conv_out_layer = torch.matmul(
            conv_out_layer, conv_kernel_layer
        )  # [B * N * num_heads, num_heads, 1]
        # [C // num_heads, ks] x [ks, 1]
        conv_out = torch.reshape(
            conv_out_layer, [bs, seqlen, -1, self.attention_head_size]
        )
        return conv_out


class ConvBertSDConv(nn.Module):
    def __init__(self, config, all_head_size, num_heads):
        super().__init__()
        self.kernel_size = config.conv_kernel_size
        self.lconv = ConvBertLightConv(config, all_head_size // num_heads)
        self.conv_attn_kernel = nn.Linear(
            all_head_size, num_heads * config.conv_kernel_size
        )
        self.separable_conv1d = SeparableConv1d(
            config.hidden_size, all_head_size, config.conv_kernel_size
        )

    def forward(self, query, value, hidden_states):
        ks = self.separable_conv1d(hidden_states)
        conv_attn_layer = ks * query
        conv_kernel_layer = self.conv_attn_kernel(
            conv_attn_layer
        )  # bs * seq_len * (num_heads * kernel_size)

        conv_kernel_layer = conv_kernel_layer.reshape(-1, self.kernel_size, 1)
        conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        conv_out = self.lconv(value, conv_kernel_layer)
        return conv_out


class ConvBertMixedAttention(nn.Module):
    def __init__(self, config):
        super(ConvBertMixedAttention, self).__init__()
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        new_num_attention_heads = int(
            config.num_attention_heads / config.head_ratio)
        if new_num_attention_heads < 1:
            config.head_ratio = config.num_attention_heads
            self.num_attention_heads = 1
        else:
            self.num_attention_heads = new_num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        config.all_head_size = self.all_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.self = ConvBertSelfAttention(config)
        self.sdconv = ConvBertSDConv(
            config, self.all_head_size, self.num_attention_heads
        )
        self.self_linear = nn.Linear(
            self.all_head_size, config.hidden_size // 2)
        self.conv_linear = nn.Linear(
            self.all_head_size, config.hidden_size // 2)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        is_calc_cka=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_value_layer = self.value(encoder_hidden_states)
        else:
            mixed_value_layer = self.value(hidden_states)

        self_out, attn_scores, one_cka = self.self(
            hidden_states,
            mixed_query_layer,
            mixed_value_layer,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            is_calc_cka=is_calc_cka,
        )

        attn_output = self_out

        conv_out = self.sdconv(
            mixed_query_layer, mixed_value_layer, hidden_states)
        bs, seqlen = hidden_states.shape[:2]

        self_out = self.self_linear(self_out.reshape(bs, seqlen, -1))
        conv_out = self.conv_linear(conv_out.reshape(bs, seqlen, -1))

        context_layer = torch.cat([self_out, conv_out], dim=-1)

        # outputs = (context_layer,) + attention_probs
        return context_layer, attn_output, attn_scores, one_cka

    def flops(self, seq_len):
        flops = 0
        # q v
        flops += self.query.in_features * self.query.out_features * seq_len
        flops += self.value.in_features * self.value.out_features * seq_len
        # self - k
        flops += self.self.key.in_features * self.self.key.out_features * seq_len
        # self - attn
        flops += self.self.key.out_features * seq_len * seq_len
        flops += self.self.key.out_features * seq_len * seq_len
        # sdconv - separable_conv1d
        # logging.info("self.sdconv.separable_conv1d.depthwise.kernel_size={}".format(self.sdconv.separable_conv1d.depthwise.kernel_size))
        # logging.info("self.sdconv.separable_conv1d.depthwise.in_channels={}".format(
        #    self.sdconv.separable_conv1d.depthwise.in_channels))
        flops += (
            seq_len
            * self.sdconv.separable_conv1d.depthwise.kernel_size[0]
            * self.sdconv.separable_conv1d.depthwise.in_channels
        )  # * self.sdconv.separable_conv1d.depthwise.out_channels // self.sdconv.separable_conv1d.depthwise.in_channels
        flops += (
            seq_len
            * 1
            * self.sdconv.separable_conv1d.pointwise.in_channels
            * self.sdconv.separable_conv1d.pointwise.out_channels
            // 1
        )
        # sdconv - ks * q
        flops += self.query.out_features * seq_len
        # sdconv - conv_attn_kernel
        flops += (
            self.sdconv.conv_attn_kernel.in_features
            * self.sdconv.conv_attn_kernel.out_features
            * seq_len
        )
        # sdconv - lconv - matmul
        flops += (
            seq_len
            * self.num_attention_heads
            * self.sdconv.lconv.conv_kernel_size
            * self.sdconv.lconv.attention_head_size
            * 1
        )
        # self_linear
        flops += seq_len * self.self_linear.in_features * self.self_linear.out_features
        # conv_linear
        flops += seq_len * self.conv_linear.in_features * self.conv_linear.out_features
        return flops


# Copied from transformers.modeling_bert.BertSelfOutput
class ConvBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ConvBertFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_ffn_hidden_size = config.max_ffn_hidden_size
        self.hidden_size_increment = config.hidden_size_increment
        self.dense1 = DynamicLinear(
            config.hidden_size, self.max_ffn_hidden_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]
        self.dense2 = DynamicLinear(
            self.max_ffn_hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense1.set_indices(
            [0, config.hidden_size - 1], [0, self.hidden_size_increment - 1]
        )
        self.dense2.set_indices(
            [0, self.hidden_size_increment - 1], [0, config.hidden_size - 1]
        )

    def add_indices(self):
        logging.info(
            'ConvBertFFN, self.dense1.get_out_dimension()={}, self.max_ffn_hidden_size={}'.format(
                self.dense1.get_out_dimension(), self.max_ffn_hidden_size
            )
        )
        if self.dense1.get_out_dimension() >= self.max_ffn_hidden_size:
            return False
        self.dense1.add_out_indices(self.hidden_size_increment)
        self.dense2.add_in_indices(self.hidden_size_increment)
        return True

    def forward(self, hidden_states, is_calc_pca=False):
        input_tensor = hidden_states
        hidden_states = self.dense1(hidden_states)
        pca_sum = 0
        out_size = self.dense1.get_out_dimension().item()
        if is_calc_pca:
            pca_sum = pca_torch(hidden_states.view(-1, out_size), (0.99,))[0]
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states, pca_sum / out_size

    def flops(self, seq_len):
        flops = 0
        flops += (
            self.dense1.get_in_dimension() * self.dense1.get_out_dimension() * seq_len
        )
        flops += (
            self.dense2.get_in_dimension() * self.dense2.get_out_dimension() * seq_len
        )
        return flops


# Copied from transformers.modeling_bert.BertAttention with Bert->Electra
class ConvBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mixed = ConvBertMixedAttention(config)
        self.output = ConvBertSelfOutput(config)
        self.pruned_heads = set()

    def flops(self, seq_len):
        flops = 0
        flops += self.mixed.flops(seq_len)
        flops += (
            seq_len * self.output.dense.in_features * self.output.dense.out_features
        )
        return flops

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        is_calc_cka=False,
    ):
        self_outputs, attn_output, attn_scores, one_cka = self.mixed(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            is_calc_cka=is_calc_cka,
        )
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output, attn_output, attn_scores, one_cka


class GLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, groups=1, bias: bool = True
    ) -> None:
        super(GLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # groups
        self.groups = groups
        self.group_in_dim = self.in_features // self.groups
        self.group_out_dim = self.out_features // self.groups
        self.weight = nn.Parameter(
            torch.Tensor(self.groups, self.group_in_dim, self.group_out_dim)
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[-1] == self.in_features
        bs = input.shape[0]
        input = input.view(-1, self.groups, self.group_in_dim).transpose(0, 1)
        outputs = (
            torch.matmul(input, self.weight)
            .transpose(0, 1)
            .contiguous()
            .view(bs, -1, self.out_features)
        )
        if self.bias is not None:
            outputs += self.bias
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->Electra
class ConvBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = 1
        self.seq_len_dim = 1
        self.attention = ConvBertAttention(config)
        self.ffn = ConvBertFFN(config)

    def flops(self, seq_len):
        flops = 0
        flops += self.attention.flops(seq_len)
        flops += self.ffn.flops(seq_len)
        return flops

    def add_indices(self):
        return self.ffn.add_indices()

    def params(self):
        params = 0
        params += sum([p.numel() for p in self.attention.parameters()])
        for m in self.ffn.modules():
            if isinstance(m, DynamicLinear):
                in_features = m.in_indices[1] - m.in_indices[0] + 1
                out_features = m.out_indices[1] - m.out_indices[0] + 1
                params += in_features * out_features + out_features
            elif isinstance(m, nn.Linear):
                params += m.in_features * m.out_features + m.out_features
            elif isinstance(m, nn.LayerNorm):
                params += m.normalized_shape[0] + m.normalized_shape[0]
        return params

    def forward(
        self, hidden_states, attention_mask=None, head_mask=None, is_calc_pca_cka=False
    ):
        # logging.info("attention_mask.shape={}".format(attention_mask.shape))
        attention_output, attn_output, attn_score, one_cka = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            is_calc_cka=is_calc_pca_cka,
        )
        layer_output, one_pca = self.ffn(
            attention_output, is_calc_pca=is_calc_pca_cka)
        return layer_output, attn_output, attn_score, one_cka, one_pca
