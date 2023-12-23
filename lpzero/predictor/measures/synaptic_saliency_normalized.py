import torch 


def synaptic_saliency_normalized(model):
    metric_array = []
    for layer in model.modules():
        if isinstance(layer, ElectraLayer):
            for sublayer in layer.intermediate.modules():
                if isinstance(sublayer, torch.nn.Linear):
                    metric_array.append(
                        torch.abs(sublayer.weight * sublayer.weight.grad)
                    )
            for sublayer in layer.output.modules():
                if isinstance(sublayer, torch.nn.Linear):
                    metric_array.append(
                        torch.abs(sublayer.weight * sublayer.weight.grad)
                    )
                    
    summed = torch.tensor(0.0)
    for j in range(len(metric_array)):
        summed += torch.nansum(metric_array[j])
    summed /= len(metric_array)
        
    return summed.detach().item()