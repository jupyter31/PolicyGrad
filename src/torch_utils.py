import math
import torch


def np2torch(x, cast_double_to_float=True, device=torch.device("cpu")):
    """
    Utility function that accepts a numpy array and does the following:
        1. Convert to torch tensor
        2. Move it to the GPU (if CUDA is available)
        3. Optionally casts float64 to float32 (torch is picky about types)
    """
    x = torch.from_numpy(x)
    if cast_double_to_float and x.dtype is torch.float64:
        x = x.float()
    x = x.to(device)
    return x


def batch_iterator(*args, batch_size=1000, shuffle=False):
    """
    Given a torch tensor or a sequence of torch tensors (which must all have
    the same first dimension), returns a generator which iterates over the
    tensor(s) in mini-batches of size batch_size.
    Pass shuffle=True to randomize the order.
    """
    if type(args) in {list, tuple}:
        multi_arg = True
        n = len(args[0])
        for i, arg_i in enumerate(args):
            assert torch.is_tensor(arg_i)
            assert len(arg_i) == n
    else:
        multi_arg = False
        n = len(args)

    indices = torch.randperm(n) if shuffle else torch.arange(n)

    n_batches = math.ceil(float(n) / batch_size)
    for batch_index in range(n_batches):
        batch_start = batch_size * batch_index
        batch_end = min(batch_size * (batch_index + 1), n)
        batch_indices = indices[batch_start:batch_end]
        if multi_arg:
            yield tuple(arg[batch_indices] for arg in args)
        else:
            yield args[batch_indices]