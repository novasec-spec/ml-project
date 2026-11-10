import torch
import torch.nn.functional as F

# sequence length = 4
# embedding dimension = 8

x = torch.randn(4, 8)

# trainable matrices

Wq = torch.randn(8, 8)
Wk = torch.randn(8, 8)
Wv = torch.randn(8, 8)

# create Q K V

Q = x @ Wq
K = x @ Wk
V = x @ Wv

# attention scores

scores = Q @ K.T

# scale

d = K.shape[-1]
scores = scores / (d ** 0.5)

# probabilities

weights = F.softmax(scores, dim=-1)

# weighted values

output = weights @ V

print("Attention weights:")
print(weights)

print("\nOutput:")
print(output)
