import torch

def log_mem(tag, tp_rank):
    torch.cuda.synchronize()
    print(
        f"[rank {tp_rank}] {tag}: "
        f"{torch.cuda.memory_allocated()/1024**2:.1f} MB alloc, "
        f"{torch.cuda.memory_reserved()/1024**2:.1f} MB reserved"
    )