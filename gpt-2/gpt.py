import numpy as np
# (1) Review Section 2 - GPT-2 Small Hyperparameters

#model hyperparameters
l_max = 1024
L = 12
H = 12
N_V = 50257

#model dimensions
d_e = 768
d_mlp = 3072
d_attn = 64
d_mid = 64
d_in = 768
d_out = 64

#total parameters
total_parameters = 124.4e6

#verify hyperparameter relationship
assert d_attn == d_e // H
assert d_mid == d_attn
assert d_in == d_e
assert H * d_mid == d_e

#display hyperparameters
print("GPT-2 Small Hyperparameters")
print("Maximum sequence length:", l_max)
print("Number of layers:", L)
print("Number of attention heads:", H)
print("Vocabulary size:", N_V)
print("Model width:", d_e)
print("MLP width:", d_mlp)
print("Attention dimension:", d_attn)
print("Per-head dimension:", d_mid)
print("Input dimension:", d_in)
print("Output dimension:", d_out)
print("Total parameters:", total_parameters / 1e6, "Million")

print("\nAll hyperparameter assertions passed!")

# (2) Implement Algorithm 1 - Token Embedding

def token_embedding(v, W_e):

    #check token ID
    assert 0 <= v < N_V

    #check embedding matrix dimensions
    assert W_e.shape == (d_e, N_V)

    #extract the embedding vector for token v
    e = W_e[:, v]

    #check output dimension
    assert e.shape == (d_e,)

    return e


#create token embedding matrix
W_e = np.random.randn(d_e, N_V).astype(np.float32)

#example token ID
v = 100

#get token embedding
e = token_embedding(v, W_e)

#display results
print("\nAlgorithm 1 - Token Embedding")
print("Embedding matrix shape:", W_e.shape)
print("Token ID:", v)
print("Embedding vector shape:", e.shape)

assert e.shape == (768,)

print("All token embedding assertions passed!")

# (3) Implement Algorithm 6 - Layer Normalization

def layer_norm(e, gamma, beta):

    #calculate mean
    m = np.sum(e) / d_e

    #calculate variance
    v = np.sum((e - m) ** 2) / d_e

    #normalize, scale and shift
    e_hat = ((e - m) / np.sqrt(v)) * gamma + beta

    return e_hat


#example input and parameters
e = np.random.randn(d_e)
gamma = np.ones(d_e)
beta = np.zeros(d_e)

#apply layer normalization
e_hat = layer_norm(e, gamma, beta)

#verify output dimension
assert e_hat.shape == (d_e,)

print("\nAlgorithm 6 - Layer Normalization")
print("Input shape:", e.shape)
print("Output shape:", e_hat.shape)
print("Mean:", np.mean(e_hat))
print("Variance:", np.var(e_hat))

print("Layer normalization assertion passed!")

# (4) Implement Algorithm 4 - ATTENTION

def attention(X, Z, W_q, W_k, W_v, b_q, b_k, b_v, Mask):

    #calculate Query, Key and Value matrices
    Q = W_q @ X + b_q[:, np.newaxis]
    K = W_k @ Z + b_k[:, np.newaxis]
    V = W_v @ Z + b_v[:, np.newaxis]

    #calculate scaled attention scores
    S = (K.T @ Q) / np.sqrt(d_attn)

    #apply causal mask
    S[Mask == 0] = -np.inf

    #apply softmax column-wise
    S = np.exp(S - np.max(S, axis=0, keepdims=True))
    S = S / np.sum(S, axis=0, keepdims=True)

    #calculate attention output
    V_tilde = V @ S

    return V_tilde


#example input and parameters
l = 5

X = np.random.randn(d_e, l)
Z = X.copy()

W_q = np.random.randn(d_attn, d_e)
W_k = np.random.randn(d_attn, d_e)
W_v = np.random.randn(d_out, d_e)

b_q = np.random.randn(d_attn)
b_k = np.random.randn(d_attn)
b_v = np.random.randn(d_out)

#create causal mask
Mask = np.triu(np.ones((l, l)))

#apply attention
V_tilde = attention(X, Z, W_q, W_k, W_v, b_q, b_k, b_v, Mask)

#verify output dimensions
assert V_tilde.shape == (d_out, l)

print("\nAlgorithm 4 - ATTENTION")
print("Input shape:", X.shape)
print("Attention output shape:", V_tilde.shape)
print("Attention assertion passed!")

# (5) Implement Algorithm 5 - MHATTENTION

def mhattention(X, W_q, W_k, W_v, b_q, b_k, b_v, W_o, b_o, Mask):

    #calculate attention for each head
    Y_heads = []

    for h in range(H):
        Y_h = attention(
            X, X,
            W_q[h], W_k[h], W_v[h],
            b_q[h], b_k[h], b_v[h],
            Mask
        )

        Y_heads.append(Y_h)

    #stack all attention heads vertically
    Y = np.vstack(Y_heads)

    #combine the outputs of all heads
    V_tilde = W_o @ Y + b_o[:, np.newaxis]

    return V_tilde


#example input and parameters
l = 5

X = np.random.randn(d_e, l)

W_q = np.random.randn(H, d_attn, d_e)
W_k = np.random.randn(H, d_attn, d_e)
W_v = np.random.randn(H, d_mid, d_e)

b_q = np.random.randn(H, d_attn)
b_k = np.random.randn(H, d_attn)
b_v = np.random.randn(H, d_mid)

W_o = np.random.randn(d_e, H * d_mid)
b_o = np.random.randn(d_e)

#create causal mask
Mask = np.triu(np.ones((l, l)))

#apply multi-head attention
V_tilde = mhattention(
    X, W_q, W_k, W_v,
    b_q, b_k, b_v,
    W_o, b_o, Mask
)

#verify output dimensions
assert V_tilde.shape == (d_e, l)

print("\nAlgorithm 5 - MHATTENTION")
print("Input shape:", X.shape)
print("Number of attention heads:", H)
print("Combined output shape:", V_tilde.shape)
print("Multi-head attention assertion passed!")