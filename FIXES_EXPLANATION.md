# Issues Found and Fixes for Gender Debiasing Implementation

## Critical Issues Identified

### 1. **Wrong LoRA Task Type** ❌

**Problem:** You used `TaskType.FEATURE_EXTRACTION` for a Masked Language Model.

**Fix:**

```python
# WRONG (your current code)
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,  # ❌ Wrong!
    ...
)

# CORRECT
lora_config = LoraConfig(
    # Don't specify task_type, or use:
    # task_type=TaskType.TOKEN_CLS for token classification-like tasks
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"],
    bias="none",
)
```

### 2. **No Token Masking Strategy** ❌

**Problem:** You're creating labels directly from input_ids without masking
tokens. This doesn't work for MLM debiasing.

**Current (Wrong):**

```python
# This doesn't mask any tokens!
pro_labels = pro_input_ids.clone()
anti_labels = anti_input_ids.clone()
```

**What Should Happen:** For gender debiasing with MLM, you need to:

1. Identify gender-specific tokens (e.g., "he", "she", "his", "her")
2. Replace them with [MASK] tokens
3. Train the model to predict them equally well in both contexts

**Example:**

```
Original Pro:  "The developer argued because [he] did not like it"
Original Anti: "The developer argued because [she] did not like it"

Masked Pro:  "The developer argued because [MASK] did not like it"
Masked Anti: "The developer argued because [MASK] did not like it"

Goal: Model should predict "he" and "she" with equal probability
```

### 3. **Incorrect Loss Function Logic** ❌

**Problem:** Your loss function computes MLM loss on unmasked tokens, which is
meaningless.

**Current Issue:**

```python
# All tokens have labels, including non-masked ones
pro_task_loss = F.cross_entropy(pro_logits_flat, pro_labels_flat, ignore_index=-100)
```

**Fix:** You need to:

1. Set labels to -100 for all non-masked tokens
2. Only compute loss on the masked gender tokens
3. Ensure consistency between pro and anti predictions

### 4. **Conceptual Mismatch** ⚠️

You're mixing two different approaches:

- **Your notebook:** Uses MLM (Masked Language Modeling)
- **Your src/debiaser/trainer.py:** Uses sequence classification

Choose ONE approach:

#### Option A: MLM Debiasing (Recommended for DistilBERT)

```python
from transformers import AutoModelForMaskedLM

# 1. Properly mask gender tokens
# 2. Compute loss only on masked positions
# 3. Minimize difference between pro/anti predictions
```

#### Option B: Classification Debiasing

```python
from transformers import AutoModelForSequenceClassification

# 1. Add classification head
# 2. Ensure pro and anti sentences get same predictions
# 3. Maintain task accuracy
```

## Recommended Fix

Here's the corrected approach for MLM-based debiasing:

### Step 1: Create a proper masking function

```python
def mask_gender_tokens(input_ids, attention_mask, tokenizer):
    """Mask gender-specific tokens."""
    gender_tokens = {
        tokenizer.encode('he', add_special_tokens=False)[0],
        tokenizer.encode('she', add_special_tokens=False)[0],
        tokenizer.encode('his', add_special_tokens=False)[0],
        tokenizer.encode('her', add_special_tokens=False)[0],
        tokenizer.encode('him', add_special_tokens=False)[0],
        # Add more gender tokens as needed
    }
    
    labels = input_ids.clone()
    masked_input_ids = input_ids.clone()
    
    # Only keep labels for gender tokens, set others to -100
    for i, token_id in enumerate(input_ids.flatten()):
        if token_id.item() not in gender_tokens:
            labels.view(-1)[i] = -100
        else:
            # Mask the gender token
            masked_input_ids.view(-1)[i] = tokenizer.mask_token_id
    
    return masked_input_ids, labels
```

### Step 2: Fix the loss function

```python
def debiasing_loss_fn(pro_logits, anti_logits, pro_labels, anti_labels, alpha=0.5):
    """
    Compute debiasing loss for MLM.
    
    Args:
        pro_logits: Logits for pro-stereotyped sentences
        anti_logits: Logits for anti-stereotyped sentences
        pro_labels: Labels (only gender tokens, others -100)
        anti_labels: Labels (only gender tokens, others -100)
        alpha: Balance between task loss and consistency loss
    """
    # 1. MLM Task Loss (only on gender tokens)
    pro_logits_flat = pro_logits.view(-1, pro_logits.size(-1))
    anti_logits_flat = anti_logits.view(-1, anti_logits.size(-1))
    pro_labels_flat = pro_labels.view(-1)
    anti_labels_flat = anti_labels.view(-1)
    
    pro_task_loss = F.cross_entropy(pro_logits_flat, pro_labels_flat, ignore_index=-100)
    anti_task_loss = F.cross_entropy(anti_logits_flat, anti_labels_flat, ignore_index=-100)
    task_loss = (pro_task_loss + anti_task_loss) / 2
    
    # 2. Consistency Loss (predictions should be similar for pro and anti)
    # Only compute on positions where labels != -100
    mask = (pro_labels != -100).float()
    
    # Get probabilities for masked positions
    pro_probs = F.softmax(pro_logits, dim=-1)
    anti_probs = F.softmax(anti_logits, dim=-1)
    
    # Compute KL divergence or MSE on masked positions
    consistency_loss = F.mse_loss(
        pro_probs * mask.unsqueeze(-1), 
        anti_probs * mask.unsqueeze(-1)
    )
    
    # 3. Total Loss
    total_loss = (1 - alpha) * task_loss + alpha * consistency_loss
    
    return {
        'total_loss': total_loss,
        'task_loss': task_loss,
        'consistency_loss': consistency_loss,
    }
```

### Step 3: Update the training loop

```python
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        pro_input_ids = batch['pro_input_ids'].to(device)
        pro_attention_mask = batch['pro_attention_mask'].to(device)
        anti_input_ids = batch['anti_input_ids'].to(device)
        anti_attention_mask = batch['anti_attention_mask'].to(device)
        
        # Mask gender tokens and create labels
        pro_masked_ids, pro_labels = mask_gender_tokens(
            pro_input_ids, pro_attention_mask, tokenizer
        )
        anti_masked_ids, anti_labels = mask_gender_tokens(
            anti_input_ids, anti_attention_mask, tokenizer
        )
        
        # Forward pass with MASKED inputs
        pro_outputs = model(input_ids=pro_masked_ids, attention_mask=pro_attention_mask)
        anti_outputs = model(input_ids=anti_masked_ids, attention_mask=anti_attention_mask)
        
        # Compute loss
        losses = debiasing_loss_fn(
            pro_outputs.logits, anti_outputs.logits, 
            pro_labels, anti_labels, alpha=0.5
        )
        
        # Backprop
        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()
        scheduler.step()
        
        epoch_loss += losses['total_loss'].item()
    
    print(f"Epoch {epoch+1} - Loss: {epoch_loss/len(train_dataloader):.4f}")
```

## Summary

Your current implementation has these critical issues:

1. ✗ Wrong LoRA task type
2. ✗ No token masking (essential for MLM debiasing)
3. ✗ Loss computed on all tokens instead of just gender tokens
4. ✗ Consistency loss doesn't focus on masked positions

Follow the recommended fixes above to properly implement MLM-based gender
debiasing.
