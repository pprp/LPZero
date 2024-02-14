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
    # head_outputs = get_head_metric_array(net)
    # act_outputs = get_act_metric_array(net, inputs, targets)

    # N = inputs.shape[0]
    # for sp in range(split_data):
    #     st = sp * N // split_data
    #     en = (sp + 1) * N // split_data

    #     if isinstance(net, ElectraModel):
    #         output = net(inputs).last_hidden_state 
    #         output.backward(torch.ones_like(output))
    #     else: # GPT-2
    #         loss, _, _, _ = net.forward(inputs[st:en, :], targets[st:en, :], mems=None)
    #         loss = loss.float().mean().type_as(loss)
    #         loss.backward()

    # lpzero
    # INPUT:(weight, grad)TREE:(element_wise_pow|l1_norm|softmax|element_wise_pow)BINARY:(element_wise_sum)
    
    # weight_outputs = compute_weight(net, inputs, targets)
    # grad_outputs = compute_gradient(net, inputs, targets)
    
    # A1, A2 = weight_outputs, grad_outputs
    # A1 = [unary_operation(a, 3) for a in A1]
    # A1 = [unary_operation(a, 10) for a in A1]
    # A2 = [unary_operation(a, 11) for a in A2]
    # A2 = [unary_operation(a, 3) for a in A2]
    
    # A = []
    # for a1, a2 in zip(A1, A2):
    #     a1 = convert_to_float(a1)
    #     a2 = convert_to_float(a2)
    #     A.append(binary_operation(a1, a2, 0))
    
    genotype = {
        'input_geno': ['weight', 'grad'],
        'op_geno': [[3, 10], [11, 3], 0]
    }
    struct = TreeStructure()
    struct.genotype = genotype 
    
    print(f'current struct: {struct}')
    
    output = struct(net, inputs, targets)
    return output 


def get_synflow_scores(args, exp_name):
    path_to_results = exp_name
    yaml_file_scores = os.path.join(path_to_results, "lpzero_scores_seed_{}.yaml".format(args.seed))
    yaml_file_cost = os.path.join(path_to_results, "lpzero_cost.yaml")
    calc_scores = not os.path.exists(yaml_file_scores)
    calc_costs = args.get_cost and not os.path.exists(yaml_file_cost)

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

                if isinstance(model, MemTransformerLM):
                    model._forward = types.MethodType(_forward_synflow_memformer, model)
                    model.forward = types.MethodType(forward_synflow_memformer, model)
                    model.crit.forward = types.MethodType(forward_crit, model.crit)

                elif isinstance(model, HfGPT2Flex):
                    model.forward = types.MethodType(forward_synflow_gpt, model)
                    model.model.lm_head.forward = types.MethodType(forward_crit, model.model.lm_head)

                B = 1
                tgt_len, mem_len, ext_len = (model_config["tgt_len"], model_config["mem_len"], model_config["ext_len"],)
                data_len = tgt_len
                data = torch.ones(data_len * B).to(device, torch.long)
                diter = LMOrderedIterator(data, B, tgt_len, device=device, ext_len=ext_len)
                if calc_scores:
                    for idx, (inp, tgt, seqlen, _) in enumerate(diter):
                        grads_abs = compute_synflow_per_weight(model, inp, tgt)
                        score = np.sum([torch.sum(g).detach().numpy() for g in grads_abs])
                        break
                    scores[config_name] = score.tolist()
                if calc_costs:
                    model.eval()
                    with torch.no_grad():
                        for _, (inp, tgt, _, _) in enumerate(diter):
                            curr_flops = get_model_flops(model, inp, tgt)
                            total_flops = np.sum([curr_flops[k] for k in ["Attn", "FFN", "Sftmax"]]).tolist()
                            break
                    costs[config_name] = 3 * total_flops
                    print(count, config_name, 'score:', scores[config_name], 'FLOPS:', costs[config_name])
                else:
                    print(count, config_name, 'score:', scores[config_name])
                count += 1

    if calc_scores:
        with open(yaml_file_scores, "w") as f:
            yaml.dump(scores, f)
    if calc_costs:
        with open(yaml_file_cost, "w") as f:
            yaml.dump(costs, f)