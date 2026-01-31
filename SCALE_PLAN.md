# Scale Plan

- **DDP**: Good for CPU/GPU nodes; simple. Used here
- **FSDP**: Only if model > GPU memory; shards paramms
- **Tensor Parallel/Pipeline Parallel**: For extremely large models


## Metrics to tack when scaling:
- Loss per step
- Effective batch size
- Wall-clock time
- GPU memory utilization

## Risks:
- Communication overhead
- Deadlocks if DDP init fails
- Gradient accumulation bugs