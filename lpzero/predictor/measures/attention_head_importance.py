import torch

from lpzero.model.flexibert.modeling_electra import ElectraLayer


# Attention Head Importance metric
def head_importance(model):
    metric_array = []
    for layer in model.modules():
        if isinstance(layer, ElectraLayer):
            for sublayer in layer.operation.operation.modules():
                if isinstance(sublayer, torch.nn.Linear):
                    if (
                        (sublayer.weight is not None)
                        and (sublayer.weight.grad is not None)
                        and sublayer.weight.shape[0] >= 128
                    ):
                        metric_array.append(
                            torch.abs(sublayer.weight.data *
                                      sublayer.weight.grad)
                        )
    summed = torch.tensor(0.0)
    for j in range(len(metric_array)):
        summed += torch.nansum(metric_array[j])

    return summed.detach().item()


def head_importance_normalized(model):
    metric_array = []
    for layer in model.modules():
        if isinstance(layer, ElectraLayer):
            for sublayer in layer.operation.operation.modules():
                if isinstance(sublayer, torch.nn.Linear):
                    if (
                        (sublayer.weight is not None)
                        and (sublayer.weight.grad is not None)
                        and sublayer.weight.shape[0] >= 128
                    ):
                        metric_array.append(
                            torch.abs(sublayer.weight.data *
                                      sublayer.weight.grad)
                        )
    summed = torch.tensor(0.0)
    for j in range(len(metric_array)):
        summed += torch.nansum(metric_array[j])
    summed /= len(metric_array)

    return summed.detach().item()
