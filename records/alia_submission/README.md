# 11L Partial RoPE + LN Scale + EMA + Late QAT + XSA4 + Reptile Meta-TTT
**val_bpb: 1.4589 
Non-record submission exploring Reptile meta-learning. Results show degradation in BPB despite reduced model size.

## Run Command
```bash
# Setup (once)
bash prepare.sh

# Train + evaluate (default seed=1337)
bash eval/eval.sh

# With specific seed
SEED=42 bash eval/eval.sh
```

### Summary 
- This submission augments a #315-style architecture with Reptile meta-learning applied during the final phase of training.
- Observed results: 
  -Increased BPB (both validation and sliding window)
  -Reduced model size (~1.5 MB)
  -Reduced effective training steps
- Conclusion: Reptile is detrimental to compression performance in this setting.

## Key Techniques

### Reptile Meta-Learning (Experimental)
- **Phase 1 (80% of training - 480s)**: Normal training with #315 architecture
- **Phase 2 (20% of training - 120s)**: Reptile meta-learning
  - Adapts MLP layers of last 25% of blocks (layers 8-10)
  - Inner loop: 3 SGD steps at LR=0.1 on random text chunks
  - Outer loop: Meta-update at LR=0.01 toward adapted weights
  - Intended to improve adaption. In practice, degrades compression metrics. 

### PR #315 Base Architecture
- **Partial RoPE (16/64 dims)**: Only first 16 of 64 head dims use rotary position encoding
- **LN Scale (1/√(layer+1))**: RMSNorm output scaled by depth factor
- **Late QAT**: Quantization-aware training enabled only in final 10% of warmdown
- **XSA4**: Cross-sequence attention on last 4 layers (layers 7-10)
- **EMA (decay=0.997)**: Exponential moving average of weights during training

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

## Attribution
- Base architecture: PR #315 (@jfprincz) - Partial RoPE, LN Scale, Late QAT, XSA4, EMA
- Reptile meta-TTT concept: PR #296 (@sseanliu) - Meta-learning for test-time adaptation
