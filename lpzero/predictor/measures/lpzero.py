import torch
import torch.nn as nn 
from lpzero.model.flexibert.modeling_electra import ElectraModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block
from lpzero.operators import unary_operation, binary_operation
from lpzero.structures.utils import convert_to_float
import transformers
import os 
# from lpzero.operators.zc_inputs import compute_weight, compute_gradient
from lpzero.structures import TreeStructure
import yaml 
import re 
import numpy as np 

from lpzero.model.model_loader import load_model_from_config
from lpzero.model.hf_gpt2.model_hf_gpt2 import HfGPT2Flex


def get_head_metric_array(net):
    head_outputs = []
    
    for layer in net.modules():
        if isinstance(layer, transformers.Conv1D) or isinstance(layer, nn.Linear):
            head_outputs.append(layer.weight)
    return head_outputs

def get_act_metric_array(net, inputs, targets):
    act_outputs = []

    def activation_hook(module, input, output):
        act_outputs.append(output.detach())

    for layer in net.modules():
        if isinstance(layer, transformers.Conv1D) or isinstance(layer, nn.Linear):
            layer.register_forward_hook(activation_hook)
        
    N = inputs.shape[0]
    for sp in range(1):
        st = sp * N // 1
        en = (sp + 1) * N // 1

        if isinstance(net, ElectraModel):
            output = net(inputs).last_hidden_state 
            output.backward(torch.ones_like(output))
        else: # GPT-2
            loss, _, _, _ = net.forward(inputs[st:en, :], targets[st:en, :], mems=None)
            loss = loss.float().mean().type_as(loss)
            loss.backward()  

    return act_outputs

def get_lpzero(net, inputs, targets=None, loss_fn=None, split_data=1, skip_grad=False):        
    # lpzero
    # INPUT:(weight, grad)TREE:(element_wise_pow|l1_norm|softmax|element_wise_pow)BINARY:(element_wise_sum)
    
    # candidate 
    # genotype = {
    #     'input_geno': ['grad', 'weight'],
    #     'op_geno': [[16, 2], [17, 1], 0]
    # }
    
    genotype = { # current best 
        'input_geno': ['weight', 'grad'],
        'op_geno': [[3, 10], [11, 3], 0]
    }
    struct = TreeStructure()
    struct.genotype = genotype 
    
    print(f'current struct: {struct}')
    output = struct(net, inputs, targets)
    return output 


def get_batch_data(train_iter, num_batches=1):
    traindata = []
    for batch, (data, target, seq_len, _) in enumerate(train_iter, start=1):
        if batch > num_batches:
            break
        traindata.append((data, target))

    inputs = torch.cat([a for a, _ in traindata], dim=-1)
    targets = torch.cat([b for _, b in traindata], dim=-1)
    return inputs, targets

def get_lpzero_scores(args, exp_name, tr_iter):
    path_to_results = exp_name
    yaml_file_scores = os.path.join(path_to_results, "lpzero_scores_seed_{}.yaml".format(args.seed))
    yaml_file_cost = os.path.join(path_to_results, "lpzero_cost.yaml")
    calc_scores = not os.path.exists(yaml_file_scores)
    calc_costs = args.get_cost and not os.path.exists(yaml_file_cost)
    
    inputs, targets = get_batch_data(tr_iter, 1)

    device = torch.device("cpu")

    files = []
    dirlist = [path_to_results]
    while len(dirlist) > 0:
        for (dirpath, dirnames, filenames) in os.walk(dirlist.pop()):
            dirlist.extend([os.path.join(dirpath, d) for d in dirnames])
            files.extend(
                map(lambda n: os.path.join(*n), zip([dirpath] * len(filenames), filenames),))

    if calc_scores or calc_costs:
        scores = {}
        costs = {}
        count = 1
        for _f in set(files):
            if "model_config.yaml" in _f:
                idx =  re.search('(config_[0-9]+)', _f).span()[0]
                job = _f[idx:]
                config_name = job.split('/')[0]
                config_name += '_' + job.split('/')[1]

                with open(_f, "r") as f:
                    model_config = yaml.full_load(f)

                model = load_model_from_config(args.model_type, model_config)
                model.n_token = model_config["n_token"]
                model.to(device)
                
                zc = get_lpzero(model, inputs, targets)

                scores[config_name] = zc
                
                print(count, config_name, 'score:', scores[config_name])
                count += 1

    if calc_scores:
        with open(yaml_file_scores, "w") as f:
            yaml.dump(scores, f)
    if calc_costs:
        with open(yaml_file_cost, "w") as f:
            yaml.dump(costs, f)