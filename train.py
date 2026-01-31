import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

# -------------------------
# Windows-safe DDP environment variables
# -------------------------
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"


def setup_ddp(rank, world_size):
    """Initialize DDP process group (CPU, Windows-safe)"""
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
    )
    torch.manual_seed(42)


def cleanup_ddp():
    dist.destroy_process_group()


def train(rank=0, world_size=1):
    """Main training loop"""

    # Only setup DDP if world_size > 1
    if world_size > 1:
        setup_ddp(rank, world_size)

    # -------------------------
    # Hyperparameters
    # -------------------------
    per_gpu_batch_size = 32
    accum_steps = 4
    lr = 1e-2
    num_steps = 20
    effective_batch_size = per_gpu_batch_size * world_size * accum_steps

    # -------------------------
    # Dataset
    # -------------------------
    x = torch.randn(1024, 10)
    y = torch.randn(1024, 1)
    dataset = TensorDataset(x, y)

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    dataloader = DataLoader(dataset, batch_size=per_gpu_batch_size, sampler=sampler)

    # -------------------------
    # Model
    # -------------------------
    model = nn.Linear(10, 1)
    if world_size > 1:
        model = DDP(model, device_ids=None)  # CPU-safe
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # -------------------------
    # CSV metrics logging (only rank 0)
    # -------------------------
    if rank == 0:
        with open("metrics.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss", "effective_batch_size", "world_size", "accum_steps"])

    # -------------------------
    # Training loop
    # -------------------------
    optimizer.zero_grad()
    step = 0
    if sampler:
        sampler.set_epoch(0)

    for xb, yb in dataloader:
        loss = loss_fn(model(xb), yb) / accum_steps
        loss.backward()

        if (step + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            if rank == 0:
                print(f"step={step} loss={loss.item() * accum_steps:.4f} effective_bs={effective_batch_size}")
                with open("metrics.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([step, loss.item() * accum_steps, effective_batch_size, world_size, accum_steps])

        step += 1
        if step >= num_steps:
            break

    if world_size > 1:
        cleanup_ddp()


if __name__ == "__main__":
    # -------------------------
    # Windows-friendly: single-process CPU training
    # Set world_size > 1 only if running on Linux/multi-GPU
    # -------------------------
    world_size = 1
    train(rank=0, world_size=world_size)