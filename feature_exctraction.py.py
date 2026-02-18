"""
ViT Inference Benchmark & WSI Simulation Tool
==============================================
Technically rigorous benchmarking of Vision Transformer inference speed
for computational pathology / whole-slide image (WSI) processing contexts.

DISCLAIMER: This tool benchmarks feature extraction throughput only.
Results do not reflect clinical performance, diagnostic accuracy, or
production pipeline latency (which includes preprocessing, I/O, postprocessing).
"""

import math
import time
import gc
from typing import Optional

import streamlit as st
import torch
import torch.nn as nn

try:
    from timm.models.vision_transformer import VisionTransformer
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants & limits
# ---------------------------------------------------------------------------
MAX_EMBED_DIM = 2048
MAX_DEPTH = 48
MAX_HEADS = 32
WARMUP_ITERS = 5

# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_vit(
    image_size: int,
    patch_size: int,
    embed_dim: int,
    depth: int,
    num_heads: int,
    mlp_ratio: float,
    global_pool: str,
) -> nn.Module:
    """
    Dynamically construct a VisionTransformer with random weights.
    Uses timm.VisionTransformer if available, otherwise falls back to
    a minimal custom implementation.

    timm global_pool / class_token interaction varies across versions and
    causes hangs or assertion errors when global_pool="cls" is combined
    with num_classes=0.  Safe fix: always build timm with global_pool=""
    (returns full token sequence) and wrap it in _TimmViTWrapper which
    applies cls-token or avg pooling explicitly after the forward pass.
    This is version-agnostic and has zero overhead.
    """
    if TIMM_AVAILABLE:
        _backbone = VisionTransformer(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=3,
            num_classes=0,           # no classification head
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            global_pool="",          # always: return full (B, N+1, D) sequence
            class_token=True,        # always keep CLS token in sequence
        )
        model = _TimmViTWrapper(_backbone, global_pool=global_pool)
    else:
        model = _MinimalViT(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            global_pool=global_pool,
        )
    # Ensure random weights (no pretrained loading)
    model.apply(_init_weights_random)
    return model


class _TimmViTWrapper(nn.Module):
    """
    Thin wrapper around a timm VisionTransformer that applies pooling
    explicitly after the backbone returns the full token sequence.

    timm with global_pool="" and class_token=True returns shape (B, N+1, D)
    where index 0 is the CLS token and indices 1..N are patch tokens.

    global_pool="cls"  → return x[:, 0]          (CLS token)
    global_pool="avg"  → return x[:, 1:].mean(1)  (mean of patch tokens only)
    """
    def __init__(self, backbone: nn.Module, global_pool: str):
        super().__init__()
        assert global_pool in ("cls", "avg"), f"Unsupported global_pool: {global_pool}"
        self.backbone    = backbone
        self.global_pool = global_pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # backbone returns (B, N+1, D) — full sequence including CLS at index 0
        out = self.backbone.forward_features(x)   # safe across timm versions
        if self.global_pool == "cls":
            return out[:, 0]           # CLS token
        else:
            return out[:, 1:].mean(1)  # mean of patch tokens


def _init_weights_random(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Minimal fallback ViT (if timm not installed)
# ---------------------------------------------------------------------------

class _MHSAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _MLP(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _MHSAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _MinimalViT(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, depth, num_heads, mlp_ratio, global_pool):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        num_patches = (image_size // patch_size) ** 2
        self.global_pool = global_pool
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if global_pool == "cls" else None
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        seq_len = num_patches + (1 if global_pool == "cls" else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.blocks = nn.Sequential(*[_Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        if self.cls_token is not None:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        if self.global_pool == "cls":
            return x[:, 0]
        return x.mean(dim=1)


# ---------------------------------------------------------------------------
# Model statistics
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """Approximate size assuming float32."""
    return count_parameters(model) * 4 / (1024 ** 2)


def estimate_flops(
    image_size: int,
    patch_size: int,
    embed_dim: int,
    depth: int,
    num_heads: int,
    mlp_ratio: float,
) -> float:
    """
    Analytical FLOPs estimate for a single image forward pass.

    Components accounted for (per transformer block):
      - Self-attention:
          QKV projection:  2 * N * embed_dim * 3*embed_dim
          Attention scores: 2 * num_heads * head_dim * N * N  (QK^T + softmax negligible)
          Attention × V:   2 * num_heads * N * N * head_dim
          Output proj:     2 * N * embed_dim * embed_dim
      - MLP:
          FC1: 2 * N * embed_dim * (embed_dim * mlp_ratio)
          FC2: 2 * N * (embed_dim * mlp_ratio) * embed_dim
      - LayerNorm: ~2 * N * embed_dim per LN (2 per block) — minor, included
      - Patch embedding (Conv2d): 2 * num_patches * patch_size^2 * 3 * embed_dim

    Returns total FLOPs (floating-point multiply-add counts × 2).
    Note: We use the convention that one MAC = 2 FLOPs.
    """
    num_patches = (image_size // patch_size) ** 2
    N = num_patches  # sequence length (ignoring cls token for simplicity)
    head_dim = embed_dim // num_heads

    # Patch embedding
    flops_patch_embed = 2 * num_patches * (patch_size ** 2) * 3 * embed_dim

    flops_per_block = 0

    # QKV projection: input (N, D) -> (N, 3D)
    flops_per_block += 2 * N * embed_dim * (3 * embed_dim)

    # Attention: QK^T  shape (H, N, head_dim) x (H, head_dim, N) => (H, N, N)
    flops_per_block += 2 * num_heads * N * N * head_dim

    # Attention × V: (H, N, N) x (H, N, head_dim) => (H, N, head_dim)
    flops_per_block += 2 * num_heads * N * N * head_dim

    # Output projection: (N, D) -> (N, D)
    flops_per_block += 2 * N * embed_dim * embed_dim

    # MLP FC1
    hidden_dim = int(embed_dim * mlp_ratio)
    flops_per_block += 2 * N * embed_dim * hidden_dim

    # MLP FC2
    flops_per_block += 2 * N * hidden_dim * embed_dim

    # LayerNorm (2 per block): approximate as 2 * N * D per LN
    flops_per_block += 4 * N * embed_dim

    total_flops = flops_patch_embed + depth * flops_per_block
    return float(total_flops)


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def run_benchmark(
    model: nn.Module,
    device: torch.device,
    image_size: int,
    batch_size: int,
    num_tiles: int,
) -> dict:
    """
    Benchmark ViT inference throughput.

    Procedure:
      1. Move model to device, set eval mode.
      2. Warm-up: run WARMUP_ITERS forward passes (not timed).
      3. Timed run: ceil(num_tiles / batch_size) forward passes.
         GPU: synchronize before/after timing window.
         CPU: time.perf_counter() is sufficient.
      4. Compute throughput = total_tiles_processed / elapsed_seconds.

    Returns dict with timing, throughput, and memory stats.
    """
    is_cuda = device.type == "cuda"

    model = model.to(device)
    model.eval()

    def make_batch(bs):
        return torch.randn(bs, 3, image_size, image_size, device=device)

    # ----- warm-up -----
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = model(make_batch(batch_size))
            if is_cuda:
                torch.cuda.synchronize(device)

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    # ----- timed benchmark -----
    num_batches = math.ceil(num_tiles / batch_size)
    tiles_processed = num_batches * batch_size

    if is_cuda:
        torch.cuda.synchronize(device)

    t_start = time.perf_counter()

    with torch.no_grad():
        for i in range(num_batches):
            x = make_batch(batch_size)
            _ = model(x)

    if is_cuda:
        torch.cuda.synchronize(device)

    t_end = time.perf_counter()
    elapsed = t_end - t_start

    throughput = tiles_processed / elapsed

    peak_mem_mb = None
    if is_cuda:
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        "elapsed_sec": elapsed,
        "tiles_processed": tiles_processed,
        "throughput_tiles_per_sec": throughput,
        "peak_memory_mb": peak_mem_mb,
        "num_batches": num_batches,
    }


# ---------------------------------------------------------------------------
# WSI simulation
# ---------------------------------------------------------------------------

def simulate_wsi(
    wsi_width: int,
    wsi_height: int,
    patch_size: int,
    overlap_pct: float,
    throughput_tiles_per_sec: float,
) -> dict:
    """
    Estimate WSI inference time given measured throughput.

    stride = patch_size * (1 - overlap_fraction)
    num_patches_x = floor((wsi_width  - patch_size) / stride) + 1
    num_patches_y = floor((wsi_height - patch_size) / stride) + 1
    total_patches = num_patches_x * num_patches_y
    estimated_time = total_patches / throughput
    """
    overlap = overlap_pct / 100.0
    stride = patch_size * (1.0 - overlap)

    if stride <= 0:
        return {"error": "Overlap must be < 100%."}
    if patch_size > wsi_width or patch_size > wsi_height:
        return {"error": "Patch size exceeds WSI dimensions."}

    nx = math.floor((wsi_width - patch_size) / stride) + 1
    ny = math.floor((wsi_height - patch_size) / stride) + 1
    total_patches = int(nx * ny)

    estimated_sec = total_patches / throughput_tiles_per_sec if throughput_tiles_per_sec > 0 else float("inf")

    return {
        "stride": stride,
        "num_patches_x": nx,
        "num_patches_y": ny,
        "total_patches": total_patches,
        "estimated_sec": estimated_sec,
        "estimated_min": estimated_sec / 60.0,
    }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ViT Inference Benchmark",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Vision Transformer Inference Benchmark")
st.caption(
    "**Disclaimer**: This tool benchmarks **feature extraction throughput only** using randomly "
    "initialized weights. Results do not reflect clinical diagnostic performance, nor do they "
    "account for I/O, preprocessing, or postprocessing latency in production pipelines."
)

if not TIMM_AVAILABLE:
    st.warning(
        "⚠️ `timm` not found. Using built-in minimal ViT implementation. "
        "Install `timm` for full architecture fidelity."
    )

# ============================================================
# SECTION 1 — Model Configuration
# ============================================================
st.header("① Model Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    image_size = st.selectbox("Image size (px)", [224, 256, 384, 512], index=0)
    patch_size = st.selectbox("Patch size (px)", [8, 14, 16, 32], index=1)

    if image_size % patch_size != 0:
        st.error(f"Image size {image_size} must be divisible by patch size {patch_size}.")
        st.stop()

with col2:
    embed_dim = st.select_slider(
        "Embed dimension",
        options=[128, 192, 256, 384, 512, 768, 1024, 1280, 1536, 2048],
        value=768,
    )
    depth = st.slider("Depth (transformer blocks)", min_value=1, max_value=MAX_DEPTH, value=12)

with col3:
    # num_heads must divide embed_dim
    valid_heads = [h for h in [1, 2, 4, 6, 8, 12, 16, 24, 32] if embed_dim % h == 0 and h <= MAX_HEADS]
    num_heads = st.selectbox("Number of heads", valid_heads, index=min(4, len(valid_heads) - 1))
    mlp_ratio = st.slider("MLP ratio", min_value=1.0, max_value=8.0, value=4.0, step=0.5)
    global_pool = st.radio("Global pooling", ["cls", "avg"], horizontal=True)

# Compute stats
num_patches = (image_size // patch_size) ** 2
seq_len = num_patches + (1 if global_pool == "cls" else 0)
flops = estimate_flops(image_size, patch_size, embed_dim, depth, num_heads, mlp_ratio)

# Build model for parameter count (on CPU, lightweight)
with st.spinner("Building model…"):
    try:
        _model_for_stats = build_vit(image_size, patch_size, embed_dim, depth, num_heads, mlp_ratio, global_pool)
        param_count = count_parameters(_model_for_stats)
        size_mb = model_size_mb(_model_for_stats)
        del _model_for_stats
        gc.collect()
        build_error = None
    except Exception as e:
        build_error = str(e)

if build_error:
    st.error(f"Model build error: {build_error}")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Parameters", f"{param_count / 1e6:.2f} M")
m2.metric("Model size (fp32)", f"{size_mb:.1f} MB")
m3.metric("FLOPs / image", f"{flops / 1e9:.2f} GFLOPs")
m4.metric("Sequence length", f"{seq_len} tokens")

if embed_dim > 1024 or depth > 24:
    st.warning("⚠️ Large configuration — may require significant GPU memory or be slow on CPU.")
if embed_dim > MAX_EMBED_DIM or depth > MAX_DEPTH:
    st.error("Configuration exceeds safety limits. Reduce embed_dim or depth.")
    st.stop()

with st.expander("ℹ️ FLOPs estimation methodology"):
    st.markdown(f"""
**Analytical estimate.**

For each transformer block, the following operations are counted (using MAC × 2 = FLOPs convention):

| Component | FLOPs |
|-----------|-------|
| QKV projection | 2 × N × D × 3D |
| QK attention scores | 2 × H × N² × d_h |
| Attention × V | 2 × H × N² × d_h |
| Output projection | 2 × N × D² |
| MLP FC1 | 2 × N × D × (D × r) |
| MLP FC2 | 2 × N × (D × r) × D |
| LayerNorm (×2) | 4 × N × D |

Where **N={num_patches}** patches, **D={embed_dim}**, **H={num_heads}** heads, **d_h={embed_dim//num_heads}**, **r={mlp_ratio}** (MLP ratio), depth={depth}.

Plus patch embedding conv: 2 × N × patch² × 3 × D.

**Total: {flops/1e9:.3f} GFLOPs/image**

*Note: Softmax, GELU, LayerNorm normalization statistics, and positional embedding additions are minor and omitted.*
    """)

# ============================================================
# SECTION 2 — Hardware Benchmarking
# ============================================================
st.divider()
st.header("② Hardware Benchmarking")

cuda_available = torch.cuda.is_available()

bcol1, bcol2, bcol3 = st.columns(3)
with bcol1:
    device_choice = st.radio(
        "Device",
        ["CPU"] + (["GPU (CUDA)"] if cuda_available else []),
        horizontal=True,
    )
    if not cuda_available and device_choice != "CPU":
        st.warning("CUDA not available. Defaulting to CPU.")
        device_choice = "CPU"

with bcol2:
    batch_size = st.select_slider("Batch size", options=[1, 2, 4, 8, 16, 32, 64, 128], value=4)

with bcol3:
    num_tiles = st.slider("Benchmark tiles (total)", min_value=10, max_value=500, value=100, step=10)

device = torch.device("cuda" if "GPU" in device_choice and cuda_available else "cpu")

# Memory warning
if device.type == "cuda":
    gpu_mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
    required_approx = size_mb * 3  # model + activations rough estimate
    if required_approx > gpu_mem * 0.8:
        st.warning(
            f"⚠️ Model (~{size_mb:.0f} MB) may exceed GPU memory ({gpu_mem:.0f} MB total). "
            "Reduce batch size or model size."
        )

run_btn = st.button("▶ Run Benchmark", type="primary", use_container_width=True)

if run_btn:
    with st.spinner(f"Warming up ({WARMUP_ITERS} iters) then benchmarking {num_tiles} tiles…"):
        try:
            bench_model = build_vit(
                image_size, patch_size, embed_dim, depth, num_heads, mlp_ratio, global_pool
            )
            results = run_benchmark(bench_model, device, image_size, batch_size, num_tiles)
            del bench_model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            st.session_state["bench_results"] = results
            st.session_state["bench_flops"] = flops

        except torch.cuda.OutOfMemoryError:
            st.error(
                "🚫 CUDA Out of Memory. Reduce batch size, embed dimension, or depth, "
                "then try again."
            )
            torch.cuda.empty_cache()
            st.stop()
        except Exception as e:
            st.error(f"Benchmark error: {e}")
            st.stop()

if "bench_results" in st.session_state:
    r = st.session_state["bench_results"]
    f = st.session_state["bench_flops"]

    throughput = r["throughput_tiles_per_sec"]
    tflops_per_sec = (throughput * f) / 1e12

    rm1, rm2, rm3, rm4 = st.columns(4)
    rm1.metric("Elapsed time", f"{r['elapsed_sec']:.2f} s")
    rm2.metric("Throughput", f"{throughput:.1f} tiles/sec")
    rm3.metric("Achieved TFLOPs/sec", f"{tflops_per_sec:.4f}")
    if r["peak_memory_mb"] is not None:
        rm4.metric("Peak GPU memory", f"{r['peak_memory_mb']:.1f} MB")
    else:
        rm4.metric("Device", "CPU")

    st.caption(
        f"Timed {r['tiles_processed']} tiles in {r['num_batches']} batches of {batch_size}. "
        f"Warmup: {WARMUP_ITERS} iterations (not included in timing)."
    )

# ============================================================
# SECTION 3 — WSI Simulation
# ============================================================
st.divider()
st.header("③ WSI Simulation")

if "bench_results" not in st.session_state:
    st.info("Run a benchmark first to enable WSI time estimation.")
else:
    wcol1, wcol2, wcol3 = st.columns(3)
    with wcol1:
        wsi_width = st.number_input("WSI width (px)", min_value=1000, max_value=200000, value=100000, step=1000)
        wsi_height = st.number_input("WSI height (px)", min_value=1000, max_value=200000, value=80000, step=1000)
    with wcol2:
        wsi_patch_size = st.number_input(
            "Tile size for WSI (px)", min_value=64, max_value=2048, value=image_size, step=64
        )
        overlap_pct = st.slider("Overlap (%)", min_value=0, max_value=75, value=0, step=5)
    with wcol3:
        st.markdown("**Measured throughput**")
        st.metric("", f"{st.session_state['bench_results']['throughput_tiles_per_sec']:.1f} tiles/sec")
        st.caption(
            "If your WSI tile size differs from the benchmarked image size, "
            "throughput may differ. Re-run benchmark with matching tile size for accuracy."
        )

    wsi_result = simulate_wsi(
        wsi_width, wsi_height, wsi_patch_size, overlap_pct,
        st.session_state["bench_results"]["throughput_tiles_per_sec"],
    )

    if "error" in wsi_result:
        st.error(wsi_result["error"])
    else:
        wm1, wm2, wm3 = st.columns(3)
        wm1.metric("Total patches", f"{wsi_result['total_patches']:,}")
        wm2.metric("Estimated time", f"{wsi_result['estimated_sec']:.1f} s")
        wm3.metric("", f"{wsi_result['estimated_min']:.2f} min")

        st.caption(
            f"Grid: {wsi_result['num_patches_x']} × {wsi_result['num_patches_y']} patches | "
            f"Effective stride: {wsi_result['stride']:.1f} px"
        )

        st.info(
            "**Note**: This estimate assumes 100% tissue coverage and ignores "
            "I/O overhead, tissue masking (background exclusion), batching pipeline latency, "
            "and network transfer time. Real-world WSI throughput is typically lower."
        )

# ============================================================
# SECTION 4 — Architectural Scaling Awareness
# ============================================================
st.divider()
st.header("④ Architectural Scaling & Deployability Notes")

st.markdown("""
### Why parameter count ≠ speed

Parameter count scales as **O(D²)** in embed dimension (dominated by attention projections and MLP layers),
but **inference time** depends on:

- **Arithmetic intensity**: How efficiently the hardware can execute matrix multiplications.
- **Memory bandwidth**: Larger models may not fit in L2/SRAM cache, causing bandwidth-bound execution.
- **Batch size**: Small batches (common in WSI streaming) underutilize GPU tensor cores.

### Attention complexity is quadratic in token count

For sequence length **N** (number of patches):

```
Attention FLOPs ∝ N²
```

Doubling image resolution quadruples N (e.g., 224→448 px with patch=16: 196→784 tokens), 
causing a **4× increase** in attention cost per block. This is the dominant scaling bottleneck for 
high-resolution pathology tiles.

### Low-end GPUs show larger throughput gaps

- High-end GPUs (A100, H100) have large HBM bandwidth and tensor cores tuned for large matrix ops.
- Consumer GPUs (RTX 3060, T4) may bottleneck on memory bandwidth at large embed dims.
- CPU inference is viable for small models but degrades steeply with depth.

### WSI processing magnifies scaling differences

A 100K × 80K px slide at 20× magnification with 256 px tiles (no overlap) yields ~120,000 patches.
A model that processes **100 tiles/sec vs 10 tiles/sec** means **20 min vs 3.3 hours** per slide.
At scale (1,000s of slides), throughput differences become clinically significant for workflow feasibility.


""")
