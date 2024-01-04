import math

import torch.nn as nn


def ratio_xavier_uniform_(tensor, ratio=2, gain=1.0):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    fan_out = ratio * fan_in
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    # Calculate uniform bounds from standard deviation
    a = math.sqrt(3.0) * std
    return nn.init._no_grad_uniform_(tensor, -a, a)


class BertBaseConfig(object):
    def __init__(self, lowercase=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.position_size = 512
        self.segment_size = 2
        self.hidden_size = 768
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 12
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = 3072
        self.num_layers = 12
        self.pad_token_id = 0
        self.sep_token_id = 102


class BertLargeConfig(object):
    def __init__(self, lowercase=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.position_size = 512
        self.segment_size = 2
        self.hidden_size = 1024
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 16
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = 4096
        self.num_layers = 24
        self.pad_token_id = 0
        self.sep_token_id = 102


class XlnetBaseConfig(object):
    def __init__(self):
        self.vocab_size = 32000
        self.position_size = 512
        self.segment_size = 2
        self.hidden_size = 768
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 12
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = 3072
        self.num_layers = 12
        self.pad_token_id = 5
        self.start_n_top = 5
        self.end_n_top = 5


class XlnetLargeConfig(object):
    def __init__(self):
        self.vocab_size = 32000
        self.position_size = 512
        self.segment_size = 2
        self.hidden_size = 1024
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 16
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = 4096
        self.num_layers = 24
        self.pad_token_id = 5
        self.start_n_top = 5
        self.end_n_top = 5


class RobertaBaseConfig(object):
    def __init__(self):
        self.vocab_size = 50265
        self.position_size = 514
        self.hidden_size = 768
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 12
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = 3072
        self.num_layers = 12
        self.pad_token_id = 1


class RobertaLargeConfig(object):
    def __init__(self):
        self.vocab_size = 50265
        self.position_size = 514
        self.hidden_size = 1024
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 16
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = 4096
        self.num_layers = 24
        self.pad_token_id = 1


class Gpt2SmallConfig(object):
    def __init__(self):
        self.vocab_size = 50257
        self.position_size = 1024
        self.hidden_size = 768
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 12
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = 3072
        self.num_layers = 12


class Gpt2mediumConfig(object):
    def __init__(self):
        self.vocab_size = 50257
        self.position_size = 1024
        self.hidden_size = 1024
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 16
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = 3072
        self.num_layers = 24


class MobileBertConfig(object):
    def __init__(self, lowercase=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.position_size = 512
        self.segment_size = 2
        self.embed_size = 128
        self.hidden_size = 512
        self.inner_hidden_size = 128
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 4
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = 512
        self.num_layers = 24
        self.pad_token_id = 0
        self.sep_token_id = 102
        self.is_tiny = True
        self.use_opt = True


class TinyBertConfig(object):
    def __init__(self, lowercase=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.position_size = 512
        self.segment_size = 2
        self.hidden_size = 312  # 768
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 12
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = 1200  # 3072
        self.num_layers = 4  # 6
        self.pad_token_id = 0
        self.sep_token_id = 102
        self.use_fit_dense = True
        self.fit_size = 768


class AutoBertConfig(object):
    def __init__(self, lowercase=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.position_size = 512
        self.segment_size = 2
        self.hidden_size = 540
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 12
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = self.hidden_size
        self.num_layers = 6
        self.pad_token_id = 0
        self.sep_token_id = 102
        self.use_fit_dense = True
        self.use_opt = False
        self.embed_size = 128
        self.fit_size = 768
        self.max_stacked_ffn = 4
        self.expansion_ratio_map = {
            '1_1': 1,
            '1_2': 1 / 2,
            '1_3': 1 / 3,
            '1_4': 1 / 4,
            '2_1': 1,
            '2_2': 1 / 2,
            '2_3': 1 / 3,
            '2_4': 1 / 4,
            '3_1': 1,
            '3_2': 1 / 2,
            '3_3': 1 / 3,
            '3_4': 1 / 4,
            '4_1': 1,
            '4_2': 1 / 2,
            '4_3': 1 / 3,
            '4_4': 1 / 4,
        }
        self.param_list = {
            'embed_fit_dense': 4.8088,
            'attn': 1.1696,
            '1_1': 0.5854,
            '1_2': 0.2935,
            '1_3': 0.1962,
            '1_4': 0.1476,
            '2_1': 1.1707,
            '2_2': 0.5870,
            '2_3': 0.3924,
            '2_4': 0.2951,
            '3_1': 1.7561,
            '3_2': 0.8805,
            '3_3': 0.5886,
            '3_4': 0.4427,
            '4_1': 2.3414,
            '4_2': 1.1740,
            '4_3': 0.7848,
            '4_4': 0.5902,
        }


class AutoBertSmallConfig(object):
    def __init__(self, lowercase=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.position_size = 512
        self.segment_size = 2
        self.hidden_size = 360
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 12
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = self.hidden_size
        self.num_layers = 6
        self.pad_token_id = 0
        self.sep_token_id = 102
        self.use_fit_dense = True
        self.use_opt = False
        self.embed_size = 128
        self.fit_size = 768
        self.max_stacked_ffn = 4
        self.expansion_ratio_map = {
            '1_1': 1,
            '1_2': 1 / 2,
            '1_3': 1 / 3,
            '1_4': 1 / 4,
            '2_1': 1,
            '2_2': 1 / 2,
            '2_3': 1 / 3,
            '2_4': 1 / 4,
            '3_1': 1,
            '3_2': 1 / 2,
            '3_3': 1 / 3,
            '3_4': 1 / 4,
            '4_1': 1,
            '4_2': 1 / 2,
            '4_3': 1 / 3,
            '4_4': 1 / 4,
        }
        self.param_list = {
            'embed_fit_dense': 4.5084,
            'attn': 0.5206,
            '1_1': 0.2606,
            '1_2': 0.1309,
            '1_3': 0.0876,
            '1_4': 0.0660,
            '2_1': 0.5213,
            '2_2': 0.2617,
            '2_3': 0.1752,
            '2_4': 0.1319,
            '3_1': 0.7819,
            '3_2': 0.3926,
            '3_3': 0.2628,
            '3_4': 0.1979,
            '4_1': 1.0426,
            '4_2': 0.5234,
            '4_3': 0.3504,
            '4_4': 0.2639,
        }


class AutoBert12Config(object):
    def __init__(self, lowercase=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.position_size = 512
        self.segment_size = 2
        self.hidden_size = 360  # 540
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 12
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = self.hidden_size
        self.num_layers = 12
        self.pad_token_id = 0
        self.sep_token_id = 102
        self.use_fit_dense = True
        self.use_opt = False
        self.embed_size = 128
        self.fit_size = 768
        self.max_stacked_ffn = 4
        self.expansion_ratio_map = {
            '1_1': 1,
            '1_2': 1 / 2,
            '1_3': 1 / 3,
            '1_4': 1 / 4,
            '2_1': 1,
            '2_2': 1 / 2,
            '2_3': 1 / 3,
            '2_4': 1 / 4,
            '3_1': 1,
            '3_2': 1 / 2,
            '3_3': 1 / 3,
            '3_4': 1 / 4,
            '4_1': 1,
            '4_2': 1 / 2,
            '4_3': 1 / 3,
            '4_4': 1 / 4,
        }
        self.param_list = {
            'embed_fit_dense': 4.5084,
            'attn': 0.5206,
            '1_1': 0.3904,
            '1_2': 0.1958,
            '1_3': 0.1309,
            '1_4': 0.0984,
            '2_1': 0.7808,
            '2_2': 0.3915,
            '2_3': 0.2617,
            '2_4': 0.1968,
            '3_1': 1.1713,
            '3_2': 0.5873,
            '3_3': 0.3926,
            '3_4': 0.2952,
            '4_1': 1.5617,
            '4_2': 0.7830,
            '4_3': 0.5234,
            '4_4': 0.3937,
        }


class AutoTinyBertConfig(object):
    def __init__(self, lowercase=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.position_size = 512
        self.segment_size = 2
        self.hidden_size = 768
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 12
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = 3072
        self.num_layers = 6
        self.pad_token_id = 0
        self.sep_token_id = 102
        self.use_fit_dense = True
        self.fit_size = 768
        self.max_stacked_ffn = 4
        self.expansion_ratio_map = {
            '1_1': 1,
            '1_2': 1 / 2,
            '1_3': 1 / 3,
            '1_4': 1 / 4,
            '2_1': 1,
            '2_2': 1 / 2,
            '2_3': 1 / 3,
            '2_4': 1 / 4,
            '3_1': 1,
            '3_2': 1 / 2,
            '3_3': 1 / 3,
            '3_4': 1 / 4,
            '4_1': 1,
            '4_2': 1 / 2,
            '4_3': 1 / 3,
            '4_4': 1 / 4,
        }
        self.param_list = {
            'embed_fit_dense': 24.4278,
            'attn': 2.3639,
            '1_1': 4.7240,
            '1_2': 2.3631,
            '1_3': 1.5762,
            '1_4': 1.1827,
            '2_1': 9.4479,
            '2_2': 4.7263,
            '2_3': 3.1524,
            '2_4': 2.3654,
            '3_1': 14.1719,
            '3_2': 7.0894,
            '3_3': 4.7286,
            '3_4': 3.5482,
            '4_1': 18.8959,
            '4_2': 9.4525,
            '4_3': 6.3048,
            '4_4': 4.7309,
        }


class SupernetConfig(object):
    def __init__(self, lowercase=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.position_size = 512
        self.segment_size = 2
        self.hidden_size = 540
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 12
        self.attn_dropout_prob = 0.1
        self.ffn_hidden_size = self.hidden_size
        self.num_layers = 6
        self.pad_token_id = 0
        self.sep_token_id = 102
        self.use_mobile_embed = True
        self.use_fit_dense = True
        self.use_opt = False
        self.embed_size = 128
        self.fit_size = 768
        self.num_stacked_ffn = 4
        self.ffn_expansion_ratio = 1


class MySupernetConfig(object):
    def __init__(self, lowercase=False, issingle=True, fixed_dimension=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.position_size = 512
        self.segment_size = 2
        self.hidden_size = 528
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 12
        self.attn_dropout_prob = 0.1
        self.max_ffn_hidden_size = int(1.5 * self.hidden_size)
        self.init_hidden_size = int(0.25 * self.hidden_size)
        self.hidden_size_increment = int(1 * self.init_hidden_size)
        self.num_layers = 12
        self.pad_token_id = 0
        self.sep_token_id = 102
        self.use_mobile_embed = True
        self.use_fit_dense = True
        self.use_opt = False
        self.embed_size = 128
        self.fit_size = 768
        self.num_stacked_ffn = 4
        self.type_each_block = 6
        self.upper_params = 15700000
        self.name = 'MySupernetConfig'

    def print_configs(self):
        # 打印对象的所有属性值
        print('===================MySupernetConfig==============')
        print('vocab_size:', self.vocab_size)
        print('position_size:', self.position_size)
        print('segment_size:', self.segment_size)
        print('hidden_size:', self.hidden_size)
        print('init_hidden_size:', self.init_hidden_size)
        print('hidden_dropout_prob:', self.hidden_dropout_prob)
        print('num_attn_heads:', self.num_attn_heads)
        print('attn_dropout_prob:', self.attn_dropout_prob)
        print('max_ffn_hidden_size:', self.max_ffn_hidden_size)
        print('hidden_size_increment:', self.hidden_size_increment)
        print('num_layers:', self.num_layers)
        print('pad_token_id:', self.pad_token_id)
        print('sep_token_id:', self.sep_token_id)
        print('use_mobile_embed:', self.use_mobile_embed)
        print('use_fit_dense:', self.use_fit_dense)
        print('use_opt:', self.use_opt)
        print('embed_size:', self.embed_size)
        print('fit_size:', self.fit_size)
        print('num_stacked_ffn:', self.num_stacked_ffn)


class MySupernetConfig_10M(object):
    def __init__(self, lowercase=False, issingle=True, fixed_dimension=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.position_size = 512
        self.segment_size = 2
        self.hidden_size = 528
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 12
        self.attn_dropout_prob = 0.1
        self.max_ffn_hidden_size = int(1.5 * self.hidden_size)
        self.init_hidden_size = int(0.25 * self.hidden_size)
        self.hidden_size_increment = int(1 * self.init_hidden_size)
        self.num_layers = 6
        self.pad_token_id = 0
        self.sep_token_id = 102
        self.use_mobile_embed = True
        self.use_fit_dense = True
        self.use_opt = False
        self.embed_size = 128
        self.fit_size = 768
        self.num_stacked_ffn = 4
        self.type_each_block = 6
        self.upper_params = 15700000
        self.name = 'MySupernetConfig'

    def print_configs(self):
        # 打印对象的所有属性值
        print('===================MySupernetConfig==============')
        print('vocab_size:', self.vocab_size)
        print('position_size:', self.position_size)
        print('segment_size:', self.segment_size)
        print('hidden_size:', self.hidden_size)
        print('init_hidden_size:', self.init_hidden_size)
        print('hidden_dropout_prob:', self.hidden_dropout_prob)
        print('num_attn_heads:', self.num_attn_heads)
        print('attn_dropout_prob:', self.attn_dropout_prob)
        print('max_ffn_hidden_size:', self.max_ffn_hidden_size)
        print('hidden_size_increment:', self.hidden_size_increment)
        print('num_layers:', self.num_layers)
        print('pad_token_id:', self.pad_token_id)
        print('sep_token_id:', self.sep_token_id)
        print('use_mobile_embed:', self.use_mobile_embed)
        print('use_fit_dense:', self.use_fit_dense)
        print('use_opt:', self.use_opt)
        print('embed_size:', self.embed_size)
        print('fit_size:', self.fit_size)
        print('num_stacked_ffn:', self.num_stacked_ffn)


class MySupernetConfig_5M(object):
    def __init__(self, lowercase=False, issingle=True, fixed_dimension=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.position_size = 512
        self.segment_size = 2
        self.hidden_size = 192
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 12
        self.attn_dropout_prob = 0.1
        self.max_ffn_hidden_size = self.hidden_size
        self.init_hidden_size = int(1 * self.hidden_size)
        self.hidden_size_increment = int(1 * self.init_hidden_size)
        if issingle == False:
            self.max_ffn_hidden_size = 2 * self.hidden_size
            if fixed_dimension == False:
                self.init_hidden_size = int(0.25 * self.hidden_size)
            self.hidden_size_increment = int(1 * self.init_hidden_size)
        self.num_layers = 6
        self.pad_token_id = 0
        self.sep_token_id = 102
        self.use_mobile_embed = True
        self.use_fit_dense = True
        self.use_opt = False
        self.embed_size = 128
        self.fit_size = 768
        self.num_stacked_ffn = 4
        self.type_each_block = 3
        self.upper_params = 5000000
        self.name = 'MySupernetConfig_5M'

    def print_configs(self):
        # 打印对象的所有属性值
        print('===================MySupernetConfig==============')
        print('vocab_size:', self.vocab_size)
        print('position_size:', self.position_size)
        print('segment_size:', self.segment_size)
        print('hidden_size:', self.hidden_size)
        print('hidden_dropout_prob:', self.hidden_dropout_prob)
        print('num_attn_heads:', self.num_attn_heads)
        print('attn_dropout_prob:', self.attn_dropout_prob)
        print('max_ffn_hidden_size:', self.max_ffn_hidden_size)
        print('init_hidden_size:', self.init_hidden_size)
        print('hidden_size_increment:', self.hidden_size_increment)
        print('num_layers:', self.num_layers)
        print('pad_token_id:', self.pad_token_id)
        print('sep_token_id:', self.sep_token_id)
        print('use_mobile_embed:', self.use_mobile_embed)
        print('use_fit_dense:', self.use_fit_dense)
        print('use_opt:', self.use_opt)
        print('embed_size:', self.embed_size)
        print('fit_size:', self.fit_size)
        print('num_stacked_ffn:', self.num_stacked_ffn)


class MobileBertForSuperNetConfig(object):
    def __init__(self, lowercase=False, issingle=True, fixed_dimension=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.position_size = 512
        self.segment_size = 2
        self.embed_size = 512
        self.hidden_size = 528
        self.inner_hidden_size = int(0.25 * self.hidden_size)
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 12
        self.attn_dropout_prob = 0.1
        self.max_ffn_hidden_size = 6 * self.inner_hidden_size
        self.init_hidden_size = int(0.25 * self.hidden_size)
        self.hidden_size_increment = int(1 * self.init_hidden_size)
        self.num_layers = 24
        self.pad_token_id = 0
        self.sep_token_id = 102
        self.is_tiny = True
        self.use_opt = True

    def print_configs(self):
        # 打印所有属性值
        print('===================MobileBertForSuperNetConfig==============')
        print('vocab_size:', self.vocab_size)
        print('position_size:', self.position_size)
        print('segment_size:', self.segment_size)
        print('embed_size:', self.embed_size)
        print('hidden_size:', self.hidden_size)
        print('inner_hidden_size:', self.inner_hidden_size)
        print('hidden_size_increment:', self.hidden_size_increment)
        print('hidden_dropout_prob:', self.hidden_dropout_prob)
        print('num_attn_heads:', self.num_attn_heads)
        print('attn_dropout_prob:', self.attn_dropout_prob)
        print('max_ffn_hidden_size:', self.max_ffn_hidden_size)
        print('num_layers:', self.num_layers)
        print('pad_token_id:', self.pad_token_id)
        print('sep_token_id:', self.sep_token_id)
        print('is_tiny:', self.is_tiny)
        print('use_opt:', self.use_opt)


class MobileBertForSuperNetConfig_5M(object):
    def __init__(self, lowercase=False, issingle=True, fixed_dimension=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.position_size = 512
        self.segment_size = 2
        self.embed_size = 512
        self.hidden_size = 192
        self.inner_hidden_size = int(0.25 * self.hidden_size)
        self.hidden_dropout_prob = 0.1
        self.num_attn_heads = 12
        self.attn_dropout_prob = 0.1
        self.max_ffn_hidden_size = self.inner_hidden_size * 4
        self.init_hidden_size = int(1 * self.hidden_size)
        self.hidden_size_increment = int(1 * self.init_hidden_size)
        if issingle == False:
            self.max_ffn_hidden_size = 8 * self.inner_hidden_size
            if fixed_dimension == False:
                self.init_hidden_size = int(0.25 * self.hidden_size)
            self.hidden_size_increment = int(1 * self.init_hidden_size)
        self.num_layers = 24
        self.pad_token_id = 0
        self.sep_token_id = 102
        self.is_tiny = True
        self.use_opt = True

    def print_configs(self):
        # 打印所有属性值
        print('===================MobileBertForSuperNetConfig==============')
        print('vocab_size:', self.vocab_size)
        print('position_size:', self.position_size)
        print('segment_size:', self.segment_size)
        print('embed_size:', self.embed_size)
        print('hidden_size:', self.hidden_size)
        print('inner_hidden_size:', self.inner_hidden_size)
        print('hidden_size_increment:', self.hidden_size_increment)
        print('hidden_dropout_prob:', self.hidden_dropout_prob)
        print('num_attn_heads:', self.num_attn_heads)
        print('attn_dropout_prob:', self.attn_dropout_prob)
        print('max_ffn_hidden_size:', self.max_ffn_hidden_size)
        print('num_layers:', self.num_layers)
        print('pad_token_id:', self.pad_token_id)
        print('sep_token_id:', self.sep_token_id)
        print('is_tiny:', self.is_tiny)
        print('use_opt:', self.use_opt)


class ConvBertConfig(object):
    def __init__(self, lowercase=False, issingle=True):
        self.vocab_size = 28996 if not lowercase else 30522
        self.embedding_size = 128
        self.hidden_size = 528
        self.num_hidden_layers = 12
        self.num_attention_heads = 12
        self.intermediate_size = 528
        self.max_ffn_hidden_size = 2 * self.hidden_size
        if issingle == False:
            # self.max_ffn_hidden_size =  self.hidden_size
            self.hidden_size_increment = int(1 * self.hidden_size)
        else:
            self.hidden_size_increment = self.hidden_size
        self.hidden_act = 'gelu'
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.type_vocab_size = 2
        self.initializer_range = 0.02
        self.layer_norm_eps = 1e-12

        self.head_ratio = 1
        self.conv_kernel_size = 9
        self.linear_groups = 1
        self.summary_type = 'first'
        self.summary_use_proj = True
        self.summary_activation = 'gelu'
        self.summary_last_dropout = 0.1
        self.pad_token_id = 0

    def print_configs(self):
        # 打印所有属性值
        print('===================ConvBertConfig==============')
        print('vocab_size:', self.vocab_size)
        print('embedding_size:', self.embedding_size)
        print('hidden_size:', self.hidden_size)
        print('num_hidden_layers:', self.num_hidden_layers)
        print('num_attention_heads:', self.num_attention_heads)
        print('intermediate_size:', self.intermediate_size)
        print('max_ffn_hidden_size:', self.max_ffn_hidden_size)
        print('hidden_act:', self.hidden_act)
        print('hidden_dropout_prob:', self.hidden_dropout_prob)
        print('hidden_size_increment:', self.hidden_size_increment)
        print('attention_probs_dropout_prob:',
              self.attention_probs_dropout_prob)
        print('max_position_embeddings:', self.max_position_embeddings)
        print('type_vocab_size:', self.type_vocab_size)
        print('initializer_range:', self.initializer_range)
        print('layer_norm_eps:', self.layer_norm_eps)
        print('head_ratio:', self.head_ratio)
        print('conv_kernel_size:', self.conv_kernel_size)
        print('linear_groups:', self.linear_groups)
        print('summary_type:', self.summary_type)
        print('summary_use_proj:', self.summary_use_proj)
        print('summary_activation:', self.summary_activation)
        print('summary_last_dropout:', self.summary_last_dropout)
        print('pad_token_id:', self.pad_token_id)


class LSTransformerConfig(object):
    # defaults come from https://github.com/facebookresearch/adaptive-span/blob/master/experiments/enwik8_small.sh
    def __init__(self, lowercase=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.attn_type = 'lsta'
        self.transformer_dim = 528
        self.dropout_prob = 0.1
        self.attention_dropout = 0.1
        self.hidden_size_increment = int(0.25 * self.transformer_dim)
        self.max_ffn_hidden_size = 2 * self.transformer_dim
        self.pooling_mode = 'MEAN'
        self.num_head = 12
        self.head_dim = self.transformer_dim // self.num_head
        self.num_landmarks = 128
        self.seq_len = 4096
        self.dim = self.transformer_dim
        self.window_size = 16
        self.fp32_attn = False
        self.debug = False
        self.conv_kernel_size = -1


class MultibranchBlockConfig(object):
    def __init__(self, lowercase=False):
        self.vocab_size = 28996 if not lowercase else 30522
        self.activation_dropout = 0.0
        self.activation_fn = 'relu'
        self.adam_betas = '(0.9, 0.98)'
        self.adam_eps = 1e-08
        self.adaptive_input = False
        self.adaptive_softmax_cutoff = None
        self.adaptive_softmax_dropout = 0
        self.all_gather_list_size = 16384
        self.arch = 'transformer_multibranch_v2_iwslt_de_en'
        self.attention_dropout = 0.0
        self.best_checkpoint_metric = 'loss'
        self.bucket_cap_mb = 25
        self.clip_norm = 0.0
        self.conv_linear = True
        self.criterion = 'label_smoothed_cross_entropy'
        self.curriculum = 0
        self.dataset_impl = None
        self.ddp_backend = 'c10d'
        self.disable_validation = False
        self.dropout = 0.2
        self.empty_cache_freq = 0
        self.encoder_attention_heads = 12
        self.encoder_branch_type = [
            'attn:1:264:12', 'lightweight:default:264:12']
        self.encoder_decoder_branch_type = (None,)
        self.encoder_embed_dim = 528
        self.encoder_embed_path = None
        self.encoder_ffn_embed_dim = 528
        self.encoder_ffn_list = [True, True, True, True, True, True]
        self.encoder_glu = (False,)
        self.encoder_kernel_size_list = [3, 7, 15, 31, 31, 31]
        self.encoder_layers = 6
        self.encoder_learned_pos = (False,)
        self.encoder_normalize_before = False
        self.fast_stat_sync = False
        self.find_unused_parameters = False
        self.fix_batches_to_gpus = False
        self.fixed_validation_seed = None
        self.input_dropout = 0.1
        self.keep_interval_updates = -1
        self.keep_last_epochs = -1
        self.label_smoothing = 0.1
        self.lazy_load = False
        self.left_pad_source = True
        self.left_pad_target = False
        self.log_format = None
        self.log_interval = 1000
        self.lr = [0.0005]
        self.lr_scheduler = 'inverse_sqrt'
        self.max_epoch = 0
        self.max_sentences = None
        self.max_sentences_valid = None
        self.max_source_positions = 1024
        self.max_target_positions = 1024
        self.max_tokens = 4096
        self.max_tokens_valid = 4096
        self.max_update = 50000
        self.maximize_best_checkpoint_metric = False
        self.memory_efficient_fp16 = False
        self.min_loss_scale = 0.0001
        self.min_lr = 1e-09
        self.no_epoch_checkpoints = False
        self.no_last_checkpoints = False
        self.no_progress_bar = True
        self.no_save = False
        self.no_save_optimizer_state = False
        self.no_token_positional_embeddings = False
        self.num_workers = 1
        self.optimizer = 'adam'
        self.optimizer_overrides = '{}'
        self.patience = -1
        self.raw_text = False
        self.required_batch_size_multiple = 8
        self.reset_dataloader = False
        self.reset_lr_scheduler = False
        self.reset_meters = False
        self.reset_optimizer = False
        self.save_interval = 1
        self.save_interval_updates = 0
        self.seed = 1
        self.sentence_avg = False
        self.share_all_embeddings = False
        self.share_decoder_input_output_embed = False
        self.skip_invalid_size_inputs_valid_test = False
        self.threshold_loss_scale = None
        self.tie_adaptive_weights = None
        self.train_subset = 'train'
        self.truncate_source = False
        self.update_freq = [32]
        self.upsample_primary = 1
        self.use_bmuf = False
        self.user_dir = None
        self.valid_subset = 'valid'
        self.validate_interval = 1
        self.warmup_init_lr = 1e-07
        self.warmup_updates = 4000
        self.weight_decay = 0.0001
        self.weight_dropout = 0.1
        self.weight_softmax = True
        self.ffn_init = ratio_xavier_uniform_
