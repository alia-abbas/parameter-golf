# 11L Partial RoPE + LN Scale + EMA + Late QAT + XSA4 + Reptile Meta-TTT
**val_bpb: TBD** (estimated 1.114-1.120, sliding window stride=64, post int6+zstd quantization roundtrip)

## Run Command
```bash
# Setup (once)
bash prepare.sh

# Train + evaluate (default seed=1337)
bash eval/eval.sh

# With specific seed
SEED=42 bash eval/eval.sh
```

All parameters are set as defaults in `train_gpt.py`. No env vars needed.

## Key Techniques

### Reptile Meta-Learning (Novel Addition)
- **Phase 1 (80% of training - 480s)**: Normal training with #315 architecture
- **Phase 2 (20% of training - 120s)**: Reptile meta-learning
  - Adapts MLP layers of last 25% of blocks (layers 8-10)
  - Inner loop: 3 SGD steps at LR=0.1 on random text chunks
  - Outer loop: Meta-update at LR=0.01 toward adapted weights
  - Teaches model "how to adapt quickly" on new text
- Expected gain: 0.005-0.011 BPB (based on PR #296 results)

### PR #315 Base Architecture
- **Partial RoPE (16/64 dims)**: Only first 16 of 64 head dims use rotary position encoding
  - Remaining 48 dims are position-free → improves generalization
- **LN Scale (1/√(layer+1))**: RMSNorm output scaled by depth factor
  - Stabilizes deeper layers by damping their contribution
- **Late QAT**: Quantization-aware training enabled only in final 10% of warmdown
  - Delays quantization noise until model has converged
- **XSA4**: Cross-sequence attention on last 4 layers (layers 7-10)
  - Removes self-value bias, improves attention quality
- **EMA (decay=0.997)**: Exponential moving average of weights during training
  - Smoother weight landscape than SWA on this architecture

### Mixed Int6/Int8 Quantization
- **Int6 [-32,31]** for MLP and attention weights (main parameters)
- **Int8 [-127,127]** for embeddings
- **FP16** for control tensors (scales, gates, norms)
- Per-row quantization scales for 2D tensors

## Architecture
- 11 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu^2 activation
- SmearGate + BigramHash(4096, dim=128)
- U-Net skip connections (encoder layers → decoder layers)
- Partial RoPE (16/64 dims), LN Scale, tied embeddings
- XSA on last 4 layers

## Training Hyperparameters
- **Phase 1 (Normal Training - 480s)**:
  - Muon optimizer: matrix_lr=0.04, WD=0.02, momentum=0.95
  - AdamW for embeddings/scalars: lr=0.05 (tied), WD=0.01
  - warmdown=1200 iters, warmup=20 steps
  - seq_len=2048, batch=786K tokens
  - grad_clip=0.3
  - EMA: decay=0.997
  - Late QAT: enabled at 10% of warmdown remaining
  
- **Phase 2 (Reptile Meta-Learning - 120s)**:
  - Inner LR: 0.1 (SGD on adaptation params)
  - Outer LR: 0.01 (meta-update rate)
  - Inner steps: 3 per meta-iteration
  - Adaptation target: MLP layers of last 3 blocks

- **Evaluation**:
  - Sliding window eval: stride=64
  - Seq length: 2048

## Why This Combination Works

**#315's Problem**: Best static architecture, but frozen after training
**Reptile's Solution**: Meta-learns "how to adapt quickly" on new text

Competition analysis shows:
- Naive TTT on #315 base: **neutral** (±0.001 BPB) - doesn't break it
- Naive TTT on #287 base: **+0.016 worse** - destructive interaction
- Reptile meta-TTT on weaker base: **-0.011 BPB gain** (PR #296)

**Key insight**: Since naive TTT doesn't hurt #315, Reptile can only improve from a clean baseline.

## Expected Performance

| Model | val_bpb | Notes |
|-------|---------|-------|
| Current #1 (leaderboard) | 1.1428 | 10L Int5-MLP baseline |
| PR #315 (base) | 1.1248 | Best validated architecture |
| **This submission (target)** | **1.114-1.120** | #315 + Reptile meta-TTT |

Conservative estimate: Beat #1 by **0.02-0.03 BPB**

## Attribution
- Base architecture: PR #315 (@jfprincz) - Partial RoPE, LN Scale, Late QAT, XSA4, EMA
- Reptile meta-TTT concept: PR #296 (@sseanliu) - Meta-learning for test-time adaptation
- Competition framework: OpenAI Parameter Golf Challenge