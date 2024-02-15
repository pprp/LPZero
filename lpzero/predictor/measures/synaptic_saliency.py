import torch

from lpzero.utils.modeling_electra import ElectraLayer, ElectraModel
from lpzero.model.hf_gpt2.model_hf_gpt2 import HfGPT2, HfGPT2Flex
from . import measure
import transformers

# Synaptic saliency metric
def synaptic_saliency(model):
    metric_array = []
    if isinstance(model, ElectraModel):
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
    elif isinstance(model, (HfGPT2, HfGPT2Flex)):
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear) or isinstance(layer, transformers.Conv1D):
                metric_array.append(
                    torch.abs(layer.weight * layer.weight.grad)
                )

    summed = torch.tensor(0.0).to('cuda')
    for j in range(len(metric_array)):
        summed += torch.nansum(metric_array[j])

    return summed.detach().item()

@measure('synaptic_saliency')
def compute_synaptic_saliency(model, inputs, targets=None, *args, **kwargs):
    if isinstance(model, ElectraModel):
        outputs = model(**inputs).last_hidden_state
        outputs.backward(torch.ones_like(outputs))
    elif isinstance(model, (HfGPT2, HfGPT2Flex)):
        loss, _, _, _ = model.forward(inputs, targets, mems=None)
        loss = loss.float().mean().type_as(loss)
        loss.backward()
    return synaptic_saliency(model)


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

    summed = torch.tensor(0.0).to('cuda')
    for j in range(len(metric_array)):
        summed += torch.nansum(metric_array[j])
    summed /= len(metric_array)

    return summed.detach().item()
