import copy

import torch

from .binary_ops import *  # noqa: F403
from .unary_ops import *  # noqa: F403

available_zc_candidates = []
_zc_candidates_impls = {}


def zc_candidates(name, bn=True, copy_net=True, force_clean=True, **impl_args):
    def make_impl(func):
        def zc_candidates_impl(model, device, *args, **kwargs):
            if copy_net:
                model = copy.copy(model)
                # net_orig.get_prunable_copy(bn=bn).to(device)
            else:
                model = model
            ret = func(model, *args, **kwargs, **impl_args)
            if copy_net and force_clean:
                import gc

                import torch

                del model
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


def get_zc_candidates(name, model, device, *args, **kwargs):
    results = _zc_candidates_impls[name](model, device, *args, **kwargs)

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
    from .zc_inputs import compute_head  # noqa: F401
    from .zc_inputs import compute_jacobs  # noqa: F401
    # from .zc_inputs import compute_softmax  # noqa: F401
    from .zc_inputs import compute_weight  # noqa: F401


load_all()
