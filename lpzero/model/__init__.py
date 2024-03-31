import logging

from .auto_bert import (
    AutoBert,
    AutoBertSingle,
    AutoTinyBert,
    AutoTinyBertSingle,
    MultiTaskAutoBert,
    MultiTaskBert,
)
from .bert import (
    Bert,
    BertAttention,
    BertClsPooler,
    BertEmbedding,
    BertFeedForwardNetwork,
    BertMaskedLMHead,
    BertSingle,
    BertTransformerBlock,
    MySupernetBertAttention,
    gelu,
)
from .configs import (
    AutoBert12Config,
    AutoBertConfig,
    AutoBertSmallConfig,
    AutoTinyBertConfig,
    BertBaseConfig,
    BertLargeConfig,
    ConvBertConfig,
    Gpt2mediumConfig,
    Gpt2SmallConfig,
    LSTransformerConfig,
    MobileBertConfig,
    MobileBertForSuperNetConfig,
    MobileBertForSuperNetConfig_5M,
    MultibranchBlockConfig,
    MySupernetConfig,
    MySupernetConfig_5M,
    MySupernetConfig_10M,
    RobertaBaseConfig,
    RobertaLargeConfig,
    SupernetConfig,
    TinyBertConfig,
    XlnetBaseConfig,
    XlnetLargeConfig,
)
from .gpt2 import Gpt2
from .mobile_bert import MobileBert, MobileBertEmbedding, MobileBertSingle
from .mysupernet import MultiTaskMySupernet, MySupernet, MySupernetSingle
from .roberta import Roberta, RobertaSingle

from .supernet import Supernet, SupernetSingle
from .tiny_bert import TinyBert, TinyBertSingle
from .xlnet import Xlnet, XlnetSingle

nas_bert_models = [
    'auto_bert',
    'auto_bert_small',
    'auto_bert_12',
    'mt_auto_bert',
    'auto_tiny_bert',
]
bert_models = [
    'bert_base',
    'bert_large',
    'mobile_bert',
    'ls_transformer',
    'tiny_bert',
    'mt_bert_base',
    'supernet',
    'mysupernet',
    'mysupernet_5M',
    'mysupernet_10M',
    'mt_mysupernet',
    'mt_mysupernet_5M',
    'mt_mysupernet_10M',
    'mobile_bert_for_supernet',
] + nas_bert_models
xlnet_models = ['xlnet_base', 'xlnet_large']
roberta_models = ['roberta_base', 'roberta_large']
gpt_models = ['gpt2_small', 'gpt2_medium']
all_models = bert_models + xlnet_models + roberta_models + gpt_models


def select_config(model_name, lowercase=None, issingle=True, fixed_dimension=False):
    if model_name in ['bert_base', 'mt_bert_base']:
        return BertBaseConfig(lowercase)
    elif model_name == 'bert_large':
        return BertLargeConfig(lowercase)
    elif model_name == 'xlnet_base':
        return XlnetBaseConfig()
    elif model_name == 'xlnet_large':
        return XlnetLargeConfig()
    elif model_name == 'roberta_base':
        return RobertaBaseConfig()
    elif model_name == 'roberta_large':
        return RobertaLargeConfig()
    elif model_name == 'gpt2_small':
        return Gpt2SmallConfig()
    elif model_name == 'gpt2_medium':
        return Gpt2mediumConfig()
    elif model_name == 'mobile_bert':
        return MobileBertConfig(lowercase)
    elif model_name == 'tiny_bert':
        return TinyBertConfig(lowercase)
    elif model_name in ['auto_bert', 'mt_auto_bert']:
        return AutoBertConfig(lowercase)
    elif model_name == 'auto_bert_small':
        return AutoBertSmallConfig(lowercase)
    elif model_name == 'auto_bert_12':
        return AutoBert12Config(lowercase)
    elif model_name == 'auto_tiny_bert':
        return AutoTinyBertConfig(lowercase)
    elif model_name == 'supernet':
        return SupernetConfig(lowercase)
    elif model_name == 'multi_branch':
        return MultibranchBlockConfig(lowercase)
    elif model_name == 'lstransformer':
        return LSTransformerConfig(lowercase)
    elif model_name == 'convbert':
        return ConvBertConfig(lowercase, issingle)
    elif model_name in ['mysupernet', 'mt_mysupernet']:
        return MySupernetConfig(lowercase, issingle, fixed_dimension)
    elif model_name in ['mysupernet_5M', 'mt_mysupernet_5M']:
        return MySupernetConfig_5M(lowercase, issingle, fixed_dimension)
    elif model_name in ['mysupernet_10M', 'mt_mysupernet_10M']:
        return MySupernetConfig_10M(lowercase, issingle, fixed_dimension)
    elif model_name == 'mobile_bert_for_supernet':
        return MobileBertForSuperNetConfig(lowercase, issingle, fixed_dimension)
    elif model_name == 'mobile_bert_for_supernet_5M':
        return MobileBertForSuperNetConfig_5M(lowercase, issingle, fixed_dimension)
    else:
        raise KeyError("Config for model '{}' is not found".format(model_name))


def select_single_model(
    model_name, lowercase, use_lm=False, issingle=True, fixed_dimension=False
):
    config = select_config(model_name, lowercase, issingle, fixed_dimension)
    if model_name in bert_models:
        if model_name in ['auto_bert', 'auto_bert_small', 'auto_bert_12']:
            return AutoBertSingle(config, use_lm)
        elif model_name == 'mobile_bert':
            return MobileBertSingle(config, use_lm)
        elif model_name == 'tiny_bert':
            return TinyBertSingle(config, use_lm)
        elif model_name == 'auto_tiny_bert':
            return AutoTinyBertSingle(config, use_lm)
        elif model_name == 'supernet':
            return SupernetSingle(config, use_lm)
        elif model_name in ['mysupernet', 'mysupernet_5M', 'mysupernet_10M']:
            return MySupernetSingle(
                config, use_lm, issingle=issingle, fixed_dimension=fixed_dimension
            )
        else:
            return BertSingle(config, use_lm)
    elif model_name in xlnet_models:
        return XlnetSingle(config)
    elif model_name in roberta_models:
        return RobertaSingle(config)


def select_model(
    model_name,
    lowercase,
    task,
    num_add_tokens=0,
    return_hid=False,
    issingle=True,
    fixed_dimension=False,
):
    config = select_config(model_name, lowercase, issingle, fixed_dimension)
    # logging.info("config.init_hidden_size=".format(config.init_hidden_size))
    # print(config.init_hidden_size)
    if model_name in bert_models:
        if model_name in ['auto_bert', 'auto_bert_small', 'auto_bert_12']:
            return AutoBert(config, task, return_hid)
        elif model_name == 'mt_auto_bert':
            return MultiTaskAutoBert(config, task, return_hid)
        elif model_name == 'mt_bert_base':
            return MultiTaskBert(config, return_hid)
        elif model_name == 'mobile_bert':
            return MobileBert(config, task, return_hid)
        elif model_name == 'tiny_bert':
            return TinyBert(config, task, return_hid)
        elif model_name == 'auto_tiny_bert':
            return AutoTinyBert(config, task, return_hid)
        elif model_name == 'supernet':
            return Supernet(config, task, return_hid)
        elif model_name in ['mysupernet', 'mysupernet_5M', 'mysupernet_10M']:
            return MySupernet(config, task, return_hid, issingle=issingle)
        elif model_name in ['mt_mysupernet', 'mt_mysupernet_5M', 'mt_mysupernet_10M']:
            # print("MultiTaskMySupernet(config, task, return_hid)")
            return MultiTaskMySupernet(config, task, return_hid, issingle=issingle)
        else:
            return Bert(config, task, return_hid)
    elif model_name in xlnet_models:
        return Xlnet(config, task)
    elif model_name in roberta_models:
        return Roberta(config, task)
    elif model_name in gpt_models:
        return Gpt2(config, num_add_tokens, use_lm=True)
