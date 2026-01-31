# Known Issues

## Multi-process DDP blocked on Windows
- Cause: PyTorch Gloo backend fails to resolve valid network devices on Windows + Python 3.14
- Symptom: makeDeviceForHostname(): unsupported gloo device
- Mitigation: Run with 'world_size = 1' locally

## torchrun libuv issue
- Cause: torchrun elastic launcher requests libuv even when not built
= Mitigation: Use single process DDP locally

## Gradient accumulation
-N/A