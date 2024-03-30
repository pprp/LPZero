import json

import torch

from lpzero.model.flexibert.modeling_electra import (
    ElectraConfig,
    ElectraLayer,
    ElectraModel,
)
from lpzero.runner.evo_search_bert import generate_inputs
from lpzero.structures.tree import TreeStructure

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

inputs = generate_inputs()

configs = []
with open('./data/BERT_benchmark.json', 'r') as f:
    configs = json.load(f)

i = 0

nas_config = configs[i]['hparams']['model_hparam_overrides']['nas_config']

gt = configs[i]['scores']['glue']
# build a new model
config = ElectraConfig(
    nas_config=nas_config,
    num_hidden_layers=len(nas_config['encoder_layers']),
    output_hidden_states=True,
)
model = ElectraModel(config)
model.to(device)
inputs.to(device)

tree = TreeStructure()

out = tree.forward_tree(inputs, model)

print(out)
