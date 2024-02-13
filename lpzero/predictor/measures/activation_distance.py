import torch
from . import measure


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
    device = net.device

    net.train()
    all_hooks = []

    for layer in net.modules():
        # convolution and transformer layers
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
            # variables/op needed for fisher computation
            layer.act = 0.0
            metric_array = []
            # function to call during backward pass (hooked on identity op at output of layer)
            def hook_factory(layer):
                def hook(module, grad_input, grad_output):
                    act = grad_output[0].detach()
                    act = act.view(act.size(0), -1)
                    x = (act > 0).float()
                    K = x @ x.t()
                    K2 = (1.0 - x) @ (1.0 - x.t())
                    
                    layer.act = act
                return hook

            hook = layer.register_backward_hook(hook_factory(layer))
            all_hooks.append(hook)

    with torch.no_grad():
        outputs = net(**inputs).last_hidden_state

    for hook in all_hooks:
        hook.remove()

    return outputs