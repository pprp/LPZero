import csv
import json

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import kendalltau, pearsonr
from lpzero.predictor.measures import *  # noqa: F403

from lpzero.model.flexibert.modeling_electra import (
    ElectraConfig,
    ElectraLayer,
    ElectraModel,
)
from transformers import ElectraTokenizerFast

configs = []
with open('./data/BERT_benchmark.json', 'r') as f:
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
def generate_bert(inputs):
    # Run metrics on all model in benchmark
    with open('./BERT_results_activation.csv', 'a') as f:
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

        gt_list = [] 
        baseline_list = []
        sd_list, sdn_list = [], []
        ss_list, ssn_list = [], []
        ad_list, adn_list = [], []
        js_list, jsn_list = [], []
        hi_list, hin_list = [], []
        hc_list, hcn_list = [], []
        hsc_list, hscn_list = [], []
        

        for i in range(50):
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
                attention_confidence(head_outputs),
                attention_confidence_normalized(head_outputs),
                attention_confidence(softmax_outputs),
                attention_confidence_normalized(softmax_outputs),
            ]

            gt_list.append(configs[i]['scores']['glue'])
            baseline_list.append(configs[0]['scores']['glue'])
            sd_list.append(synaptic_diversity(model))
            sdn_list.append(synaptic_diversity_normalized(model))
            ss_list.append(synaptic_saliency(model))
            ssn_list.append(synaptic_saliency_normalized(model))
            ad_list.append(activation_distance(activation_outputs))
            adn_list.append(activation_distance_normalized(activation_outputs))
            js_list.append(jacobian_score(model))
            jsn_list.append(jacobian_score_cosine(model))
            hi_list.append(head_importance(model))
            hin_list.append(head_importance_normalized(model))
            hc_list.append(attention_confidence(head_outputs))
            hcn_list.append(attention_confidence_normalized(head_outputs))
            hsc_list.append(attention_confidence(softmax_outputs))
            hscn_list.append(attention_confidence_normalized(softmax_outputs))
            

            writer.writerow(row)
            f.flush()

            print(str(configs[i]['id']))

        # plot for ac, adn 
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        sns.set_context("paper", font_scale=1.5)
        plt.figure(figsize=(8, 6))
        plt.scatter(adn_list, gt_list, s=10)
        plt.xlabel("Activation Distance Normalized")
        plt.ylabel("GLUE Score")
        plt.savefig("adn.png")
        

# predict the score of each candidate answer

# calculate the rank consistency



if __name__ == '__main__':
    inputs = generate_inputs()
    generate_bert(inputs)