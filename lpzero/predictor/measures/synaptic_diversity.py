import torch

from lpzero.utils.modeling_electra import ElectraLayer, ElectraModel
from lpzero.model.hf_gpt2.model_hf_gpt2 import HfGPT2, HfGPT2Flex
from . import measure
import transformers 


# Synaptic Diversity metric
def synaptic_diversity(model):
    metric_array = []
    if isinstance(model, ElectraModel):
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
    elif isinstance(model, (HfGPT2, HfGPT2Flex)):
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear) or isinstance(layer, transformers.Conv1D):
                if (layer.weight is not None) and (layer.weight.grad is not None):
                    metric_array.append(
                        torch.abs(
                            torch.norm(layer.weight, 'nuc')
                            * torch.norm(layer.weight.grad, 'nuc')
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


@measure('synaptic_diversity')
def compute_synaptic_diversity(model, inputs, targets, *args, **kwargs):
    loss, _, _, _ = model(inputs, targets, mems=None)
    loss = loss.float().mean().type_as(loss)
    loss.backward()
    output =  synaptic_diversity(model)
    return output 