import csv
import json

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import kendalltau, pearsonr

from lpzero.model.flexibert.modeling_electra import (
    ElectraConfig,
    ElectraLayer,
    ElectraModel,
    ElectraTokenizerFast,
)

configs = []
with open('./nas_configs.json', 'r') as f:
    configs = json.load(f)


# generate inputs
def generate_inputs():
    # Target dataset is openwebtex
    dataset = load_dataset('openwebtext')
    tokenizer = ElectraTokenizerFast.from_pretrained(
        'google/electra-small-discriminator'
    )

    def encode(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length')

    tokenized_dataset = dataset.map(encode, batched=True, num_proc=32)
    tokenized_dataset.set_format(
        type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask']
    )

    # get sample tokenized batch from dataset
    dataloader = torch.utils.data.DataLoader(dataset['train'], batch_size=128)
    inputs = tokenizer(
        next(iter(dataloader))['text'],
        truncation=True,
        padding='max_length',
        return_tensors='pt',
    )
    return inputs


# generate model BERT
def generate_bert():
    # Run metrics on all model in benchmark
    with open('BERT_initialization_ablation.csv', 'a') as f:
        writer = csv.writer(f)

        header = [
            'ID',
            'GLUE Score',
            'Synaptic Diversity',
            'Synaptic Diversity Normalized',
            'Synaptic Saliency',
            'Synaptic Saliency Normalized',
            'Activation Distance',
            'Activation Distance Normalized',
            'Jacobian Score',
            'Jacobian Score Normalized',
            'Number of Parameters',
            'Head Importance',
            'Head Importance Normalized',
            'Head Confidence',
            'Head Confidence Normalized',
            'Head Softmax Confidence',
            'Head Softmax Confidence Normalized',
        ]
        writer.writerow(header)
        f.flush()

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        for i in range(500):
            np.random.seed(0)
            torch.manual_seed(0)

            nas_config = configs[i]['hparams']['model_hparam_overrides']['nas_config']

            config = ElectraConfig(
                nas_config=nas_config,
                num_hidden_layers=len(nas_config['encoder_layers']),
                output_hidden_states=True,
            )
            model = ElectraModel(config)
            model.to(device)
            inputs.to(device)

            # Hooks to get outputs at different layers
            activation_outputs = []

            def activation_hook(module, input, output):
                activation_outputs.append(output)

            for layer in model.modules():
                if isinstance(layer, ElectraLayer):
                    sublayer = (
                        layer.intermediate.intermediate_act_fn.register_forward_hook(
                            activation_hook
                        )
                    )

            head_outputs = []

            def head_hook(module, input, output):
                head_outputs.append(output)

            # Initialize hooks
            for layer in model.modules():
                if isinstance(layer, ElectraLayer):
                    sublayer = layer.operation.operation
                    if hasattr(sublayer, 'query'):
                        sublayer.query.register_forward_hook(head_hook)
                    if hasattr(sublayer, 'key'):
                        sublayer.key.register_forward_hook(head_hook)
                    if hasattr(sublayer, 'value'):
                        sublayer.value.register_forward_hook(head_hook)
                    if hasattr(sublayer, 'input'):
                        sublayer.input.register_forward_hook(head_hook)
                    if hasattr(sublayer, 'weight'):
                        sublayer.weight.register_forward_hook(head_hook)

            softmax_outputs = []

            def softmax_hook(module, input, output):
                softmax_outputs.append(output)

            for layer in model.modules():
                if isinstance(layer, ElectraLayer):
                    sublayer = layer.operation.operation
                    if hasattr(sublayer, 'softmax'):
                        sublayer.softmax.register_forward_hook(softmax_hook)

            # Run gradient with respect to ones
            model.zero_grad()
            output = model(**inputs).last_hidden_state
            output.backward(torch.ones_like(output))

            row = [
                configs[i]['id'],
                configs[i]['scores']['glue'],
                synaptic_diversity(model),
                synaptic_diversity_normalized(model),
                synaptic_saliency(model),
                synaptic_saliency_normalized(model),
                activation_distance(activation_outputs),
                activation_distance_normalized(activation_outputs),
                jacobian_score(model),
                jacobian_score_cosine(model),
                num_parameters(model),
                head_importance(model),
                head_importance_normalized(model),
                attention_condfidence(head_outputs),
                attention_condfidence_normalized(head_outputs),
                attention_condfidence(softmax_outputs),
                attention_condfidence_normalized(softmax_outputs),
            ]

            writer.writerow(row)
            f.flush()

            print(str(configs[i]['id']))


# predict the score of each candidate answer

# calculate the rank consistency
