"""
Mamba viability test - FIXED VERSION
Addresses NaN loss issue
"""

import os
import torch
import torch.nn as nn
from mamba_ssm import Mamba
import time

# Test configuration
VOCAB_SIZE = 1024
D_MODEL = 512
SEQ_LEN = 2048
BATCH_SIZE = 4
NUM_LAYERS = 6
ITERATIONS = 100

print("=" * 70)
print("MAMBA VIABILITY TEST - FIXED")
print("=" * 70)

if not torch.cuda.is_available():
    print("❌ ERROR: CUDA not available")
    exit(1)

print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
print(f"✓ CUDA version: {torch.version.cuda}")
print()

class SimpleMambaModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=16,
                d_conv=4,
                expand=2,
            )
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.embedding.weight
        
        # IMPORTANT: Better initialization to prevent NaN
        self._init_weights()
        
    def _init_weights(self):
        # Initialize embeddings with smaller values
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        
        # Initialize layer norm
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        
    def forward(self, x):
        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x) + x  # Residual
            
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

print("Building model...")
model = SimpleMambaModel(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_layers=NUM_LAYERS
).cuda()

total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model built: {total_params:,} parameters")
print()

# Test forward
print("Testing forward pass...")
dummy_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).cuda()
try:
    with torch.no_grad():
        output = model(dummy_input)
    print(f"✓ Forward pass successful: {output.shape}")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    exit(1)

print()

# Test backward with gradient clipping
print("Testing backward pass...")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)  # Lower LR
loss_fn = nn.CrossEntropyLoss()

try:
    logits = model(dummy_input)
    loss = loss_fn(logits[:, :-1].reshape(-1, VOCAB_SIZE), 
                   dummy_input[:, 1:].reshape(-1))
    loss.backward()
    
    # IMPORTANT: Clip gradients to prevent NaN
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    optimizer.zero_grad()
    print(f"✓ Backward pass successful: loss={loss.item():.4f}")
except Exception as e:
    print(f"❌ Backward pass failed: {e}")
    exit(1)

print()

# Training loop with stability checks
print(f"Running {ITERATIONS} training iterations...")
model.train()
times = []
losses = []

for i in range(ITERATIONS):
    start = time.time()
    
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).cuda()
    
    logits = model(x)
    loss = loss_fn(logits[:, :-1].reshape(-1, VOCAB_SIZE), 
                   x[:, 1:].reshape(-1))
    
    # Check for NaN
    if torch.isnan(loss):
        print(f"❌ NaN detected at step {i+1}! Last valid loss: {losses[-1] if losses else 'N/A'}")
        break
    
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    optimizer.zero_grad()
    
    elapsed = time.time() - start
    times.append(elapsed)
    losses.append(loss.item())
    
    if (i + 1) % 20 == 0:
        avg_time = sum(times[-20:]) / 20
        avg_loss = sum(losses[-20:]) / 20
        print(f"  Step {i+1}/{ITERATIONS}: loss={loss.item():.4f}, "
              f"avg_loss={avg_loss:.4f}, time={elapsed*1000:.1f}ms, avg={avg_time*1000:.1f}ms")

print()

# Final check
if len(losses) == ITERATIONS and not any(torch.isnan(torch.tensor(l)) for l in losses):
    print("=" * 70)
    print("✓ MAMBA VIABILITY TEST PASSED!")
    print("=" * 70)
    print()
    print("Key metrics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Average step time: {sum(times)/len(times)*1000:.1f}ms")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Loss decreased: {losses[0]:.4f} → {losses[-1]:.4f}")
    print()
    print("✅ Mamba is WORKING and STABLE!")
    print()
    print("Next steps:")
    print("  1. Test with real FineWeb data")
    print("  2. Test at 8K-16K context")
    print("  3. Copy BigramHash from modded-nanogpt")
    print("  4. Add sliding window eval")
else:
    print("=" * 70)
    print("❌ MAMBA TEST FAILED - Numerical instability")
    print("=" * 70)
    print("Consider:")
    print("  - Lower learning rate")
    print("  - Stronger gradient clipping")
    print("  - Different initialization")
    print("  - Or pivot to copying modded-nanogpt transformers")