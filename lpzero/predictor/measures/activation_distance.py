import torch
from . import measure
import transformers 


from lpzero.utils.modeling_electra import ElectraLayer, ElectraModel
from lpzero.model.hf_gpt2.model_hf_gpt2 import HfGPT2, HfGPT2Flex


# Activation Distance metric
def activation_distance(outputs):
    metric_array = []
    for output in outputs:
        output = output[0].view(output.size(1), -1)
        x = (output > 0).float()
        K = x @ x.t()
        K2 = (1.0 - x) @ (1.0 - x.t())
        metric_array.append(K + K2)

    summed = torch.tensor(0.0).to('cuda')
    for j in range(len(outputs)):
        summed += torch.nansum(metric_array[j])

    return summed.detach().item()


def activation_distance_normalized(outputs):
    metric_array = []
    for output in outputs:
        output = output[0].view(output.size(1), -1)
        x = (output > 0).float()
        K = x @ x.t()
        K2 = (1.0 - x) @ (1.0 - x.t())
        metric_array.append(K + K2)

    summed = torch.tensor(0.0).to('cuda')
    for j in range(len(outputs)):
        summed += torch.nansum(metric_array[j])
    summed /= len(metric_array)

    return summed.detach().item()


@measure("activation_distance")
def compute_act_dist(net, inputs, targets=None, loss_fn=None, mode=None,
                     split_data=1):
    if hasattr(net, 'device'):
        device = net.device
    if hasattr(inputs, 'device'):
        device = inputs.device
    

    net.train()
    activation_outputs = []
    
    def activation_hook(module, input, output):
        activation_outputs.append(output)
    
    for layer in net.modules():
        if isinstance(layer, ElectraLayer):
            sublayer = layer.intermediate.intermediate_act_fn.register_forward_hook(activation_hook)
        elif isinstance(layer, torch.nn.Linear) or isinstance(layer, transformers.Conv1D):
            sublayer = layer.register_forward_hook(activation_hook)


    with torch.no_grad():
        if isinstance(net, ElectraModel):
            outputs = net(**inputs).last_hidden_state
        elif isinstance(net, (HfGPT2, HfGPT2Flex)):
            loss, _, _, _ = net.forward(inputs, targets, mems=None)

    return activation_distance(activation_outputs)