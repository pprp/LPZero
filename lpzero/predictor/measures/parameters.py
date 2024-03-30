from . import measure

@measure('num_parameters')
def num_parameters(model, *args, **kwargs):
    return sum(p.numel() for p in model.parameters())
