import copy

import torch

from .binary_ops import *  # noqa: F403
from .unary_ops import *  # noqa: F403

available_zc_candidates = []
_zc_candidates_impls = {}


def zc_candidates(name, bn=True, copy_net=True, force_clean=True, **impl_args):
    def make_impl(func):
        def zc_candidates_impl(net, device, *args, **kwargs):
            if copy_net:
                net = copy.copy(net)
                # net_orig.get_prunable_copy(bn=bn).to(device)
            else:
                net = net
            ret = func(net, *args, **kwargs, **impl_args)
            if copy_net and force_clean:
                import gc

                import torch

                del net
                torch.cuda.empty_cache()
                gc.collect()
            return ret

        global _zc_candidates_impls
        if name in _zc_candidates_impls:
            raise KeyError(f'Duplicated zc_candidates! {name}')
        available_zc_candidates.append(name)
        _zc_candidates_impls[name] = zc_candidates_impl
        return func

    return make_impl


def get_zc_candidates(name, net, device, *args, **kwargs):
    # a = torch.cuda.memory_allocated(device=torch.device('cuda:0'))
    results = _zc_candidates_impls[name](net, device, *args, **kwargs)
    # b = torch.cuda.memory_allocated(device=torch.device('cuda:0'))
    # print("Memory allocated: ", b - a, " bytes")

    # force clean
    import gc

    torch.cuda.empty_cache()
    gc.collect()

    return results


def get_zc_function(name):
    return _zc_candidates_impls[name]


def load_all():
    # from .zc_inputs import compute_hessian  # noqa: F401
    from .zc_inputs import compute_activation  # noqa: F401
    from .zc_inputs import compute_gradient  # noqa: F401
    from .zc_inputs import compute_virtual_grad  # noqa: F401
    from .zc_inputs import compute_weight  # noqa: F401


load_all()
