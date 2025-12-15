# Scaled dot-product attention
# Implement scaled dot‑product attention in Python (no libraries except NumPy)

import numpy as np

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    # 1. Find the maximum value along the chosen axis
    x_max = np.max(x, axis=axis, keepdims=True)

    # 2. Subtract the max from x before exponentiation
    #    This prevents overflow when x has large values
    e_x = np.exp(x - x_max)

    # 3. Normalize by dividing by the sum of exponentials
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, seq_len_q, d_k)
    K: (batch, seq_len_k, d_k)
    V: (batch, seq_len_k, d_v)
    mask: optional (batch, seq_len_q, seq_len_k) with 0 for keep, -1e9 for block

    Returns:
      output: (batch, seq_len_q, d_v)
      attention_weights: (batch, seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]

    # Step 1: Compute raw scores
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

    # Step 2: Apply mask if provided
    if mask is not None:
        scores = scores + mask

    # Step 3: Softmax to get attention weights
    attention_weights = softmax(scores, axis=-1)

    # Step 4: Weighted sum of values
    output = np.matmul(attention_weights, V)

    return output, attention_weights


# Example Run
x = np.array([2.0, 1.0, 0.1])
probs = softmax(x)
print("Softmax:", probs)
print("Sum of probs:", np.sum(probs))  # should be 1.0


# Example: batch=1, seq_len=4, d_k=d_v=8
np.random.seed(42)
Q = np.random.rand(1, 4, 8)
K = np.random.rand(1, 4, 8)
V = np.random.rand(1, 4, 8)

out, attn = scaled_dot_product_attention(Q, K, V)

print("Output shape:", out.shape)        # (1, 4, 8)
print("Attention shape:", attn.shape)    # (1, 4, 4)
print("Attention weights (row sums):", np.sum(attn, axis=-1))  # should be ~1


