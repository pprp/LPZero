from lpzero.model.flexibert.modeling_electra import (
    ElectraConfig,
    ElectraLayer,
    ElectraModel,
)
import csv
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import load_dataset
from scipy.stats import kendalltau, pearsonr
from tqdm import tqdm
from transformers import ElectraTokenizerFast

warnings.simplefilter(action='ignore', category=FutureWarning)

from lpzero.predictor.measures import *  # noqa: F403

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
        param_list = []
        sd_list, sdn_list = [], []
        ss_list, ssn_list = [], []
        ad_list, adn_list = [], []
        js_list, jsn_list = [], []
        hi_list, hin_list = [], []
        hc_list, hcn_list = [], []
        hsc_list, hscn_list = [], []

        for i in tqdm(range(500)):
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
            param_list.append(num_parameters(model))

            writer.writerow(row)
            f.flush()

            print(str(configs[i]['id']))

        def plot_correlations():
            headers = [
                ('Synaptic Diversity', sd_list),
                ('Synaptic Saliency', ss_list),
                ('Activation Distance', ad_list),
                ('Jacobian Score', js_list),
                ('Head Importance', hi_list),
                ('Head Confidence', hc_list),
                ('Head Softmax Confidence', hsc_list),
                ('Number of Parameters', param_list),
            ]

            # Set up the matplotlib figure
            num_plots = len(headers)
            cols = 2  # You can change this to your desired number of columns
            rows = num_plots // cols + (num_plots % cols > 0)

            sns.set_theme(style='whitegrid', context='talk', palette='Dark2')
            sns.set_context('paper', font_scale=1.2)

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
            fig.tight_layout(pad=5.0)

            for index, (header, data_list) in enumerate(headers):
                ax = axes[index // cols, index % cols]

                # Replace data_list's infs or nans with zero
                data_list = np.nan_to_num(
                    data_list, nan=0.0, posinf=0.0, neginf=0.0)

                tau, _ = kendalltau(data_list, gt_list)
                rho, _ = pearsonr(data_list, gt_list)

                sns.regplot(
                    x=gt_list, y=data_list, scatter_kws={'s': 10}, fit_reg=True, ax=ax
                )
                ax.set_ylabel(header)
                ax.set_xlabel('GLUE Score')
                ax.set_title(f'τ: {tau:.3f} ρ: {rho:.3f}')

                # Optionally, adjust the limits and other aesthetics
                ax.set_xlim([min(gt_list), max(gt_list)])
                ax.set_ylim([min(data_list), max(data_list)])

                sns.scatterplot(
                    x=gt_list,
                    y=data_list,
                    size=3,
                    edgecolor=None,
                    hue_norm=(0, 7),
                    legend=False,
                    ax=ax,
                )

            # Hide any unused subplots
            for i in range(index + 1, rows * cols):
                fig.delaxes(axes.flatten()[i])

            plt.savefig('combined_correlation.png')

        plot_correlations()


if __name__ == '__main__':
    inputs = generate_inputs()
    generate_bert(inputs)
