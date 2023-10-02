import torch

def nested_to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, list) or isinstance(x, tuple):
        return [nested_to_device(x_i, device) for x_i in x]
    elif isinstance(x, dict):
        return {k: nested_to_device(v, device) for k, v in x.items()}
    else:
        return x