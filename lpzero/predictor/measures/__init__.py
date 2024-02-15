
available_measures = []
_measure_impls = {}


def measure(name, bn=True, copy_net=True, force_clean=True, **impl_args):
    def make_impl(func):
        def measure_impl(net_orig, device, *args, **kwargs):
            if copy_net:
                net = net_orig.get_prunable_copy(bn=bn).to(device)
            else:
                net = net_orig.to(device)
            ret = func(net, *args, **kwargs, **impl_args)
            if copy_net and force_clean:
                import gc
                import torch

                del net
                torch.cuda.empty_cache()
                gc.collect()
            return ret

        global _measure_impls
        if name in _measure_impls:
            raise KeyError(f"Duplicated measure! {name}")
        available_measures.append(name)
        _measure_impls[name] = measure_impl
        return func

    return make_impl


def calc_measure(name, net, device, *args, **kwargs):
    return _measure_impls[name](net, device, *args, **kwargs)


def load_all():
    from . import grad_norm
    from . import snip
    from . import grasp
    from . import fisher
    from . import jacob_cov
    from . import plain
    from . import jacob_cov_relu
    from . import parameters
    from . import synaptic_diversity
    from . import synaptic_saliency
    from . import activation_distance
    from . import attention_confidence
    from . import attention_head_importance
    from . import jacobian_score
    from . import eznas 


# TODO: should we do that by default?
load_all()
