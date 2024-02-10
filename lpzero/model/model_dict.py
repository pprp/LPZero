# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Availability dictionaries of implemented Transformer-based classes.
"""

# Huggingface's Open AI GPT-2
from lpzero.model.hf_gpt2.config_hf_gpt2 import (HfGPT2Config, HfGPT2SearchConfig,
                                                      HfGPT2FlexConfig, HfGPT2FlexSearchConfig)
from lpzero.model.hf_gpt2.model_hf_gpt2 import HfGPT2, HfGPT2Flex

# Huggingface's Transformer-XL
from lpzero.model.hf_transfo_xl.config_hf_transfo_xl import (HfTransfoXLConfig,
                                                                  HfTransfoXLSearchConfig)
from lpzero.model.hf_transfo_xl.model_hf_transfo_xl import HfTransfoXL

# NVIDIA's Memory Transformer
from lpzero.model.mem_transformer.config_mem_transformer import (MemTransformerLMConfig,
                                                                      MemTransformerLMSearchConfig)
from lpzero.model.mem_transformer.model_mem_transformer import MemTransformerLM


# Analytical parameters formulae
from lpzero.model.model_utils.analytical_params_formulae import (get_params_hf_gpt2_formula,
                                                                      get_params_hf_gpt2_flex_formula,
                                                                      get_params_hf_transfo_xl_formula,
                                                                      get_params_mem_transformer_formula)

MODELS = {
    'hf_gpt2': HfGPT2,
    'hf_gpt2_flex': HfGPT2Flex,
    'hf_transfo_xl': HfTransfoXL,
    'mem_transformer': MemTransformerLM
}

MODELS_CONFIGS = {
    'hf_gpt2': HfGPT2Config,
    'hf_gpt2_flex': HfGPT2FlexConfig,
    'hf_transfo_xl': HfTransfoXLConfig,
    'mem_transformer': MemTransformerLMConfig
}

MODELS_SEARCH_CONFIGS = {
    'hf_gpt2': HfGPT2SearchConfig,
    'hf_gpt2_flex': HfGPT2FlexSearchConfig,
    'hf_transfo_xl': HfTransfoXLSearchConfig,
    'mem_transformer': MemTransformerLMSearchConfig
}

MODELS_PARAMS_FORMULAE = {
    'hf_gpt2': get_params_hf_gpt2_formula,
    'hf_gpt2_flex': get_params_hf_gpt2_flex_formula,
    'hf_transfo_xl': get_params_hf_transfo_xl_formula,
    'mem_transformer': get_params_mem_transformer_formula
}