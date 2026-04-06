import gc

try:
    import torch
except Exception:
    torch = None


def clear_iteration_memory(_: int, __, ___) -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    gc.collect()


def clear_model_memory() -> None:
    clear_iteration_memory(0, None, None)
