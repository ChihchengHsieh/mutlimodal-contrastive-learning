import torch, gc

def clean_memory_get_device() -> str:
    gc.collect()
    # torch.cuda.memory_summary(device=None, abbreviated=False)

    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    print(f"This notebook will running on device: [{device.upper()}]")

    if use_gpu:
        torch.cuda.empty_cache()
    return device
