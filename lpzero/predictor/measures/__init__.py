


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
    from .activation_distance import activation_distance, activation_distance_normalized
    from .attention_confidence import attention_confidence, attention_confidence_normalized
    from .attention_head_importance import head_importance, head_importance_normalized
    from .jacobian_covariance import covariance
    from .jacobian_score import jacobian_score, jacobian_score_cosine
    from .parameters import num_parameters
    from .synaptic_diversity import synaptic_diversity, synaptic_diversity_normalized
    from .synaptic_saliency import synaptic_saliency, synaptic_saliency_normalized


# TODO: should we do that by default?
load_all()
