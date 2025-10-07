# Improved Debiasing Loss Function with KL Divergence

## Issue with Current Implementation

Your current consistency loss multiplies probabilities by a sparse mask, which
dilutes the signal:

```python
# CURRENT (Less Effective)
mask = (pro_labels != -100).float().unsqueeze(-1)
masked_pro_probs = pro_probs * mask  # Lots of zeros
masked_anti_probs = anti_probs * mask
consistency_loss = F.mse_loss(masked_pro_probs, masked_anti_probs)
```

## Improved Implementation

```python
def debiasing_loss_fn_improved(pro_logits, anti_logits, pro_labels, anti_labels, alpha=0.5):
    """
    Improved debiasing loss with proper KL divergence on masked positions.
    """
    # 1. MLM Task Loss (unchanged - this is correct)
    pro_logits_flat = pro_logits.view(-1, pro_logits.size(-1))
    anti_logits_flat = anti_logits.view(-1, anti_logits.size(-1))
    pro_labels_flat = pro_labels.view(-1)
    anti_labels_flat = anti_labels.view(-1)
    
    pro_task_loss = F.cross_entropy(pro_logits_flat, pro_labels_flat, ignore_index=-100)
    anti_task_loss = F.cross_entropy(anti_logits_flat, anti_labels_flat, ignore_index=-100)
    task_loss = (pro_task_loss + anti_task_loss) / 2
    
    # 2. IMPROVED Consistency Loss
    # Only compute on positions where we have masked tokens
    mask = (pro_labels != -100)  # Boolean mask [batch, seq_len]
    
    if mask.any():
        # Extract only the masked position logits
        masked_pro_logits = pro_logits[mask]  # [num_masked_tokens, vocab_size]
        masked_anti_logits = anti_logits[mask]  # [num_masked_tokens, vocab_size]
        
        # Symmetric KL divergence (Jensen-Shannon style)
        # KL(pro || anti) + KL(anti || pro) / 2
        consistency_loss = (
            F.kl_div(
                F.log_softmax(masked_pro_logits, dim=-1),
                F.softmax(masked_anti_logits.detach(), dim=-1),  # detach to avoid double backprop
                reduction='batchmean'
            ) + 
            F.kl_div(
                F.log_softmax(masked_anti_logits, dim=-1),
                F.softmax(masked_pro_logits.detach(), dim=-1),
                reduction='batchmean'
            )
        ) / 2
    else:
        consistency_loss = torch.tensor(0.0, device=pro_logits.device)
    
    # 3. Total Loss
    total_loss = (1 - alpha) * task_loss + alpha * consistency_loss
    
    return {
        'total_loss': total_loss,
        'task_loss': task_loss.item(),
        'consistency_loss': consistency_loss.item(),
        'alpha': alpha
    }
```

## Why This is Better

1. **Focuses on Relevant Positions**: Only computes divergence on actual masked
   tokens
2. **Proper Distribution Distance**: KL divergence is the right metric for
   comparing probability distributions
3. **Symmetric**: Ensures both pro and anti are equally penalized for divergence
4. **No Dilution**: Doesn't waste computation on zero-padded positions

## Comparison

| Metric              | Current MSE             | Improved KL                    |
| ------------------- | ----------------------- | ------------------------------ |
| Signal Strength     | Weak (diluted by zeros) | Strong (only masked positions) |
| Distribution Metric | MSE (suboptimal)        | KL divergence (proper)         |
| Gradient Quality    | Noisy                   | Clean                          |
| Effectiveness       | Medium                  | High                           |

## Drop-in Replacement

Just replace your `debiasing_loss_fn` function with `debiasing_loss_fn_improved`
in your notebook!
