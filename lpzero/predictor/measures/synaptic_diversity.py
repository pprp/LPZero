import torch

from modeling_electra import ElectraLayer


# Synaptic Diversity metric
def synaptic_diversity(model):
    metric_array = []
    for layer in model.modules():
        if isinstance(layer, ElectraLayer):
            for sublayer in layer.operation.modules():
                if isinstance(sublayer, torch.nn.Linear):
                    if (sublayer.weight is not None) and (
                        sublayer.weight.grad is not None
                    ):
                        metric_array.append(
                            torch.abs(
                                torch.norm(sublayer.weight, 'nuc')
                                * torch.norm(sublayer.weight.grad, 'nuc')
                            )
                        )
    summed = torch.tensor(0.0).to('cuda')
    for j in range(len(metric_array)):
        summed += torch.nansum(metric_array[j])

    return summed.detach().item()


def synaptic_diversity_normalized(model):
    metric_array = []
    for layer in model.modules():
        if isinstance(layer, ElectraLayer):
            for sublayer in layer.operation.modules():
                if isinstance(sublayer, torch.nn.Linear):
                    if (sublayer.weight is not None) and (
                        sublayer.weight.grad is not None
                    ):
                        metric_array.append(
                            torch.abs(
                                torch.norm(sublayer.weight, 'nuc')
                                * torch.norm(sublayer.weight.grad, 'nuc')
                            )
                        )

    summed = torch.tensor(0.0).to(model.device)
    for j in range(len(metric_array)):
        summed += torch.nansum(metric_array[j])
    summed /= len(metric_array)

    return summed.detach().item()
