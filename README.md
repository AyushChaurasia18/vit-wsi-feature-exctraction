# ViT Inference Benchmark & WSI Simulation Tool

A technically rigorous Streamlit application for benchmarking Vision Transformer (ViT) 
inference throughput and simulating whole-slide image (WSI) processing constraints.

---

## Setup

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate.bat         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For CUDA-accelerated PyTorch, install from https://pytorch.org/get-started/locally/ 
matching your CUDA version **before** running the above, e.g.:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Run

```bash
streamlit run feature_exctraction.py
```

---

## Application Structure

| Section | Purpose |
|---------|---------|
| ① Model Configuration | Build ViT with user-defined architecture; display params, FLOPs |
| ② Hardware Benchmarking | Measure real forward-pass throughput on CPU or GPU |
| ③ WSI Simulation | Estimate full-slide inference time from measured throughput |

---


## FLOPs Estimation Methodology

FLOPs are estimated **analytically** from architecture parameters.

The convention used: **1 MAC (multiply-accumulate) = 2 FLOPs**.

### Per transformer block

| Operation | Formula |
|-----------|---------|
| QKV projection | 2 × N × D × 3D |
| QK^T (attention scores) | 2 × H × N² × d_h |
| Attention × V | 2 × H × N² × d_h |
| Output projection | 2 × N × D × D |
| MLP FC1 | 2 × N × D × (r × D) |
| MLP FC2 | 2 × N × (r × D) × D |
| LayerNorm (×2) | 4 × N × D |

Where:
- **N** = number of patches = (image_size / patch_size)²
- **D** = embed_dim
- **H** = num_heads
- **d_h** = D / H (head dimension)
- **r** = mlp_ratio

### Plus patch embedding

```
2 × N × patch_size² × 3 × D
```

### Omissions (intentional)

- Softmax numerics (negligible vs. matmul)
- GELU activation (elementwise, ~1 FLOP/element)
- Positional embedding addition (elementwise)
- LayerNorm variance/mean computation (minor)

---

## Benchmarking Methodology

### Procedure

1. **Model construction**: Random weight initialization via `trunc_normal_` — no pretrained 
   checkpoints loaded. This is valid for throughput benchmarking since forward-pass speed 
   is architecture-dependent, not weight-dependent.

2. **Warm-up** (`WARMUP_ITERS = 5`): Discarded iterations to allow GPU to reach steady-state 
   clock frequency, fill CUDA caches, and JIT-compile any lazy operations.

3. **Timed window**:
   - GPU: `torch.cuda.synchronize()` is called **before** starting the timer and **after** 
     the final forward pass to ensure the CUDA kernel queue is flushed.
   - CPU: `time.perf_counter()` is used directly (no async ops on CPU).
   - All forward passes run under `torch.no_grad()` to disable gradient tracking.

4. **Throughput**: `tiles / elapsed_wall_clock_seconds`

5. **Memory**: `torch.cuda.max_memory_allocated()` after resetting stats at benchmark start.

### What this does NOT measure

- Disk I/O (reading tile images)
- JPEG/PNG decompression
- Tissue masking / background exclusion
- GPU→CPU transfer for downstream tasks
- Multi-GPU parallelism

---

## WSI Simulation

Given measured throughput **T** (tiles/sec):

```
stride           = patch_size × (1 - overlap_fraction)
num_patches_x    = floor((WSI_width  - patch_size) / stride) + 1
num_patches_y    = floor((WSI_height - patch_size) / stride) + 1
total_patches    = num_patches_x × num_patches_y
estimated_time   = total_patches / T
```

**Assumption**: All patches are processed (100% tissue coverage). Real pipelines apply 
tissue detection to skip background tiles, reducing actual patch count by 30–70% 
depending on specimen type.

---

## Known Limitations

- FLOPs estimate does not account for flash attention implementations, which can have 
  different effective FLOPs/sec due to tiling strategies.
- `timm` versions differ in exact architecture details; results may vary slightly across versions.
- CPU benchmarks are single-threaded by default in PyTorch unless `OMP_NUM_THREADS` is set.
- CUDA timing precision is ~microsecond level; very fast small-model benchmarks may show 
  higher variance.

---

## Disclaimer

This tool is intended for **infrastructure planning and research exploration only**.
Results do not reflect clinical performance, diagnostic accuracy, or production system latency.
