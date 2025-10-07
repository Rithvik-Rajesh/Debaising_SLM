# Analysis & Performance Improvements for Gender Debiasing

## ✅ Current Implementation Status

Your implementation is **much better** now! The bracket-based masking approach
is correct and should work well.

---

## 🔍 Potential Issues & Fixes

### 1. **Consistency Loss May Be Too Weak** ⚠️

**Issue**: You're multiplying probabilities by a mask that's mostly zeros, which
can make the consistency loss very small and ineffective.

**Current Code**:

```python
mask = (pro_labels != -100).float().unsqueeze(-1)  # [batch, seq_len, 1]
masked_pro_probs = pro_probs * mask
masked_anti_probs = anti_probs * mask
consistency_loss = F.mse_loss(masked_pro_probs, masked_anti_probs)
```

**Problem**: This computes MSE over ALL positions (including zeros), diluting
the signal.

**Better Approach**:

```python
# Only compute loss on actual masked positions
mask = (pro_labels != -100)  # [batch, seq_len]

# Extract only the masked position logits
if mask.any():
    masked_pro_logits = pro_logits[mask]  # [num_masks, vocab_size]
    masked_anti_logits = anti_logits[mask]  # [num_masks, vocab_size]
    
    # Compute KL divergence or MSE only on these positions
    pro_probs = F.softmax(masked_pro_logits, dim=-1)
    anti_probs = F.softmax(masked_anti_logits, dim=-1)
    consistency_loss = F.kl_div(
        F.log_softmax(masked_pro_logits, dim=-1),
        anti_probs,
        reduction='batchmean'
    )
else:
    consistency_loss = torch.tensor(0.0, device=pro_logits.device)
```

---

### 2. **Alpha Balance May Need Tuning** ⚠️

**Current**: `alpha=0.5` gives equal weight to task and consistency loss.

**Recommendation**: Start with higher consistency weight early, then decrease:

```python
# Dynamic alpha scheduling
def get_alpha(epoch, total_epochs):
    """Start with high consistency, gradually focus on task loss"""
    return 0.7 - (0.3 * epoch / total_epochs)  # 0.7 -> 0.4
```

---

### 3. **Gradient Clipping Missing** ⚠️

Add gradient clipping to prevent exploding gradients:

```python
# After loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

### 4. **No Validation During Training** ⚠️

You only evaluate at the end. Add periodic validation:

```python
# Add validation every N steps
if (step + 1) % eval_steps == 0:
    val_metrics = evaluate(model, val_dataloader)
    print(f"Step {step}: Val Loss = {val_metrics['loss']:.4f}")
```

---

## 🚀 Performance Improvements

### 1. **Optimize DataLoader** 🔥

**Add `num_workers` for parallel data loading**:

```python
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=16, 
    shuffle=True,
    num_workers=4,  # Parallel data loading
    pin_memory=True  # Faster GPU transfer
)
```

---

### 2. **Mixed Precision Training** 🔥🔥

**Faster training with ~2x speedup on GPU**:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_dataloader:
    with autocast():  # Mixed precision
        pro_outputs = model(input_ids=pro_input_ids, attention_mask=pro_attention_mask)
        anti_outputs = model(input_ids=anti_input_ids, attention_mask=anti_attention_mask)
        losses = debiasing_loss_fn(pro_outputs.logits, anti_outputs.logits, 
                                   pro_labels, anti_labels, alpha=0.5)
    
    scaler.scale(losses['total_loss']).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

---

### 3. **Increase Batch Size** 🔥

**Current**: batch_size=16\
**Recommendation**: Try 32 or 64 if you have enough memory

Larger batches = better gradient estimates and faster training

---

### 4. **Adjust Learning Rate** 🔥

**Current**: `lr=5e-5`\
**Try**: `lr=2e-4` with warmup

LoRA can handle higher learning rates than full fine-tuning.

---

### 5. **More LoRA Modules** 🔥

**Current**: Only `["q_lin", "v_lin"]`\
**Better**: Add all attention projections

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],  # All attention
    bias="none",
)
```

Or even include FFN:

```python
target_modules=["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"]
```

---

### 6. **Longer Training** 🔥

**Current**: 3 epochs\
**Recommendation**: 5-10 epochs

Debiasing is a subtle task and may need more iterations.

---

### 7. **Add Regularization** 🔥

Prevent overfitting to counterfactual pairs:

```python
# Add dropout to LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.2,  # Increased from 0.1
    ...
)
```

---

### 8. **Better Consistency Loss (KL Divergence)** 🔥🔥

**Current**: MSE between probability distributions\
**Better**: KL divergence (proper measure for distributions)

```python
# Symmetric KL divergence (more robust)
def js_divergence(p, q):
    """Jensen-Shannon divergence (symmetric KL)"""
    m = 0.5 * (p + q)
    return 0.5 * F.kl_div(p.log(), m, reduction='batchmean') + \
           0.5 * F.kl_div(q.log(), m, reduction='batchmean')

consistency_loss = js_divergence(pro_probs, anti_probs)
```

---

### 9. **Add Bias Metrics During Training** 🔥

Track bias reduction in real-time:

```python
def compute_bias_score(model, tokenizer, device):
    """Compute average bias on test sentences"""
    test_sentences = [
        "The nurse said [GENDER] would be back soon.",
        "The developer argued because [GENDER] did not like the design.",
    ]
    
    bias_scores = []
    he_id = tokenizer.encode('he', add_special_tokens=False)[0]
    she_id = tokenizer.encode('she', add_special_tokens=False)[0]
    
    for sent in test_sentences:
        sent = sent.replace('[GENDER]', tokenizer.mask_token)
        inputs = tokenizer(sent, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            mask_pos = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            probs = F.softmax(logits[0, mask_pos, :], dim=-1)
            he_prob = probs[0, he_id].item()
            she_prob = probs[0, she_id].item()
            
            # Compute bias as log ratio
            bias = abs(np.log(he_prob / she_prob)) if she_prob > 0 else 10.0
            bias_scores.append(bias)
    
    return np.mean(bias_scores)

# Use during training
if epoch % 1 == 0:
    bias_score = compute_bias_score(model, tokenizer, device)
    print(f"  Average Bias Score: {bias_score:.4f} (lower is better)")
```

---

### 10. **Save Best Model** 🔥

```python
best_bias_score = float('inf')

for epoch in range(epochs):
    # ... training ...
    
    bias_score = compute_bias_score(model, tokenizer, device)
    
    if bias_score < best_bias_score:
        best_bias_score = bias_score
        model.save_pretrained("output/best_debiased_model")
        print(f"✅ Saved best model (bias score: {bias_score:.4f})")
```

---

## 📊 Recommended Hyperparameters

```python
# Training config
BATCH_SIZE = 32  # or 64 if you have GPU memory
LEARNING_RATE = 2e-4  # Higher for LoRA
EPOCHS = 8
WARMUP_RATIO = 0.1
ALPHA_START = 0.7  # High consistency weight initially
ALPHA_END = 0.4    # Lower at end
MAX_GRAD_NORM = 1.0

# LoRA config
LORA_R = 32  # Increased from 16
LORA_ALPHA = 64  # 2x r
LORA_DROPOUT = 0.15
TARGET_MODULES = ["q_lin", "k_lin", "v_lin", "out_lin"]
```

---

## 🎯 Priority Fixes (Do These First)

1. **Fix consistency loss** - Use KL divergence on masked positions only
2. **Add gradient clipping** - Prevents training instability
3. **Increase target modules** - Add k_lin, out_lin for better coverage
4. **Add mixed precision** - 2x speedup on GPU
5. **Track bias metrics** - Monitor debiasing effectiveness during training

---

## 📈 Expected Results

With these improvements:

- **Training speed**: 2-3x faster with mixed precision + parallel loading
- **Debiasing quality**: Better with KL divergence + more LoRA modules
- **Stability**: Improved with gradient clipping + better alpha scheduling
- **Monitoring**: Real-time bias tracking shows progress

---

## 🔧 Quick Win Implementation

Here's a minimal set of changes for immediate improvement:

```python
# 1. Better consistency loss
def debiasing_loss_fn(pro_logits, anti_logits, pro_labels, anti_labels, alpha=0.5):
    # Task loss (unchanged)
    pro_logits_flat = pro_logits.view(-1, pro_logits.size(-1))
    anti_logits_flat = anti_logits.view(-1, anti_logits.size(-1))
    pro_labels_flat = pro_labels.view(-1)
    anti_labels_flat = anti_labels.view(-1)
    
    pro_task_loss = F.cross_entropy(pro_logits_flat, pro_labels_flat, ignore_index=-100)
    anti_task_loss = F.cross_entropy(anti_logits_flat, anti_labels_flat, ignore_index=-100)
    task_loss = (pro_task_loss + anti_task_loss) / 2
    
    # IMPROVED: Consistency loss only on masked positions
    mask = (pro_labels != -100)
    if mask.any():
        masked_pro_logits = pro_logits[mask]
        masked_anti_logits = anti_logits[mask]
        consistency_loss = F.kl_div(
            F.log_softmax(masked_pro_logits, dim=-1),
            F.softmax(masked_anti_logits, dim=-1),
            reduction='batchmean'
        ) + F.kl_div(
            F.log_softmax(masked_anti_logits, dim=-1),
            F.softmax(masked_pro_logits, dim=-1),
            reduction='batchmean'
        )
        consistency_loss = consistency_loss / 2  # Average both directions
    else:
        consistency_loss = torch.tensor(0.0, device=pro_logits.device)
    
    total_loss = (1 - alpha) * task_loss + alpha * consistency_loss
    
    return {
        'total_loss': total_loss,
        'task_loss': task_loss,
        'consistency_loss': consistency_loss,
    }

# 2. Add gradient clipping in training loop
for batch in train_dataloader:
    # ... forward pass ...
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ADD THIS
    optimizer.step()
    scheduler.step()
```

This alone should significantly improve your results! 🚀
