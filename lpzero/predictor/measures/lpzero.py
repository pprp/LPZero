import torch
import torch.nn as nn 
from . import measure
from lpzero.model.flexibert.modeling_electra import ElectraModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block
from lpzero.operators import unary_operation, binary_operation
from lpzero.structures.utils import convert_to_float
import transformers


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

@measure("lpzero", bn=True)
def get_lpzero(net, inputs, targets, loss_fn, split_data=1, skip_grad=False):
    net.train()
    for param in net.parameters():
        param.grad = None
        
    head_outputs = get_head_metric_array(net)
    act_outputs = get_act_metric_array(net, inputs, targets)

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

    A1, A2 = head_outputs, act_outputs
    A1 = [unary_operation(a, 14) for a in A1]
    A1 = [unary_operation(a, 8) for a in A1]
    A2 = [unary_operation(a, 15) for a in A2]
    A2 = [unary_operation(a, 10) for a in A2]
    
    A = []
    for a1, a2 in zip(A1, A2):
        a1 = convert_to_float(a1)
        a2 = convert_to_float(a2)
        A.append(binary_operation(a1, a2, 0))
    return A

