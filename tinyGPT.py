import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
n_embd = 384
n_blocks = 6
n_heads = 6
dropout = 0.2
# ------------


import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters (assumed to be defined elsewhere in your code)
n_embd = 384      # embedding dimension (input channels)
block_size = 256  # maximum sequence length (for masking)
dropout = 0.2     # dropout rate


class Head(nn.Module):
    def __init__(self, head_size, head_idx, block_idx, use_cache=True):
        """
        Args:
            head_size: dimension for the keys, queries, and values.
            use_cache: if True, use the KV cache variant.
        """
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Lower triangular matrix for causal masking (size: block_size x block_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        # Flag to turn on/off caching. By default, caching is off.
        self.use_cache = use_cache
        # Internal storage for cached keys and values.
        self.cache_k = None  # shape: (B, T_cached, head_size)
        self.cache_v = None  # shape: (B, T_cached, head_size)
        self.head_size = head_size
        self.head_idx = head_idx
        self.block_idx = block_idx

    def forward_no_cache(self, x):
        """
        Standard attention computation without KV cache.
        
        Complexity: O(B * T^2)
          - x: (B, T, n_embd)
          - k, q, v: each (B, T, head_size)
          - Attention scores computed over all T tokens.
        """
        B, T, C = x.shape  # C should equal n_embd
        k = self.key(x)     # (B, T, head_size)
        q = self.query(x)   # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        #if self.unique_print():
            #print(k.shape)
            #print(k)

        # Compute scaled dot-product attention scores
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, T)
        # Apply causal mask: each token can only attend to previous tokens (including itself)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        out = wei @ v      # (B, T, head_size)
        return out

    def forward_with_cache(self, x):
        """
        Attention computation using internal KV cache.
        
        In cached mode, we assume that x is a (B, T, n_embd) tensor, but during
        generation T should be 1 (i.e. only the new token is provided).
        
        Complexity per step: O(B * T_cached)
          - T_cached is the length of the cached sequence (grows over time).
          - The new token's query attends to all previous cached keys.
        """
        B, T, C = x.shape
        # We expect T to be 1 for cached generation; if not, we extract the last token.
        if T > 1:
            x_new = x[:, -1:, :]  # (B, 1, n_embd)
        else:
            x_new = x  # (B, 1, n_embd)

        # Compute projections for the new token.
        k_new = self.key(x_new)   # (B, 1, head_size)
        q = self.query(x_new)       # (B, 1, head_size)
        v_new = self.value(x_new)   # (B, 1, head_size)

        # Append the new keys/values to the internal cache.
        if self.cache_k is None:
            self.cache_k = k_new    # (B, 1, head_size)
            self.cache_v = v_new    # (B, 1, head_size)
        else:
            self.cache_k = torch.cat([self.cache_k, k_new], dim=1)  # (B, T_cached+1, head_size)
            self.cache_v = torch.cat([self.cache_v, v_new], dim=1)  # (B, T_cached+1, head_size)

            # TODO check this?
            # If the cache length is greater than or equal to block_size, remove the first token.
            if self.cache_k.shape[1] > block_size:
                self.cache_k = self.cache_k[:, 1:, :]  # Remove the token at T=0
                self.cache_v = self.cache_v[:, 1:, :]

        # Use the cached keys and values for computing attention.
        # q: (B, 1, head_size); k: (B, T_total, head_size); v: (B, T_total, head_size)
        k = self.cache_k
        v = self.cache_v
    
        '''
        T_total = k.size(1)
        scaling = C ** -0.5  # same scaling factor
        wei = q @ k.transpose(-2, -1) * scaling  # (B, 1, T_total)
        # Create a causal mask for the new token: it can only attend to tokens up to its position.
        mask = self.tril[:1, :T_total]  # (1, T_total) since q has T=1
        wei = wei.masked_fill(mask == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # (B, 1, T_total)
        wei = self.dropout(wei)
        out = wei @ v  # (B, 1, head_size)
        '''

        # old-style attention
        # Compute scaled dot-product attention scores
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, T)
        # Apply causal mask: each token can only attend to previous tokens (including itself)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        out = wei @ v

        #if self.unique_print():
        #    print(out)
        return out

    def forward(self, x):
        """
        Unified forward method that selects the correct implementation.
        
        Args:
            x: Input tensor of shape (B, T, n_embd). For cached generation, T is expected
               to be 1 (or only the last token is used).
            use_cache (bool, optional): If provided, overrides self.use_cache.
        
        Returns:
            out: The attention output.
        """
        if self.use_cache:
            #if self.unique_print():
            #    print("using cache")
            return self.forward_with_cache(x)
        else:
            #if self.unique_print():
            #    print("NOT using cache")
            return self.forward_no_cache(x)

    def unique_print(self):
        return self.head_idx == 0 and self.block_idx == 0

    def reset_cache(self):
        """
        Resets the internal KV cache.
        """
        self.cache_k = None
        self.cache_v = None



class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size, block_idx, use_cache):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, head_idx, block_idx, use_cache) for head_idx in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff(x)


class Block(nn.Module):
    def __init__(self, n_embd, block_idx, use_cache, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, block_idx, use_cache)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# our tiny GPT model
class TinyGPTModel(nn.Module):
    def __init__(self, vocab_size, use_cache=False):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, block_idx, use_cache, n_head=n_heads) for block_idx in range(n_blocks)]
        )
        self.ln_fin = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)

        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)
        x = self.ln_fin(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]

            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

