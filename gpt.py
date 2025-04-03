import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, embed_dim):
        super().__init__()
        # input x: (B, C, E)
        # B: batch size, C: context length, E: embed dimension
        # split input into heads
        self.heads = heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim//heads
        assert (self.head_dim * heads == embed_dim)
        
        self.key   = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.value = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.linear = nn.Linear(self.embed_dim, self.embed_dim)
    
    def forward(self, key, value, query, mask=None):
        # let key_length = query_length = value_length = context_length
        batch_size = key.shape[0]
        key_length, value_length, query_length = key.shape[1], value.shape[1], query.shape[1]

        # reshape to multiple heads (B, C, H, D)
        keys    = key.reshape(batch_size, key_length, self.heads, self.head_dim)
        values  = value.reshape(batch_size, value_length, self.heads, self.head_dim)
        queries = query.reshape(batch_size, query_length, self.heads, self.head_dim)
        
        # find keys, values, and queries
        keys    = self.key(keys)
        values  = self.value(values)
        queries = self.query(queries)

        # calculate values
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) # (B, H, C, C)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-inf")) # -inf evaluates to zero after softmax

        attention = torch.softmax(energy / self.embed_dim**0.5, dim=3) # (B, H, C, C)

        # apply attention mask on values and reshape (B, C, E)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(batch_size, query_length, self.embed_dim)

        return self.linear(out)
    
class FeedForward(nn.Module):
    def __init__(self, embed_dim, embed_dim_multiple=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim, embed_dim*embed_dim_multiple),
                                 nn.ReLU(),
                                 nn.Linear(embed_dim*embed_dim_multiple, embed_dim),
                                 nn.Dropout(dropout))
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, embed_dim, heads, context_len, device='cpu'):
        super().__init__()
        self.mask = torch.tril(torch.ones(context_len, context_len)).to(device)
        self.sa = MultiHeadAttention(heads=heads, embed_dim=embed_dim)
        self.ffwd = FeedForward(embed_dim=embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    
    def forward(self, x):
        norm_x = self.ln1(x)
        x = x + self.sa(key=norm_x, value=norm_x, query=norm_x, mask=self.mask)
        norm_x = self.ln2(x)
        x = x + self.ffwd(norm_x)
        return x


class GPT(nn.Module):
    def __init__(self, context_len, vocab_size, embed_dim, heads, num_layers, device):
        super().__init__()
        self.context_len = context_len
        self.device = device
        self.word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.position_embedding = nn.Embedding(num_embeddings=context_len, embedding_dim=embed_dim)
        self.block = nn.Sequential(*[Block(embed_dim=embed_dim, heads=heads, context_len=context_len, device=device) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)


    def forward(self, x):
        # expand such that 
        positions = torch.arange(0, self.context_len).to(self.device)
        out = self.word_embedding(x) + self.position_embedding(positions)
        out = self.block(out)
        logits = self.lm_head(self.ln_f(out))
        return logits

    def calculate_loss(self, logits, targets):
        B, C, E = logits.shape
        logits = logits.view(B*C, E)
        targets = targets.view(B*C)
        loss = F.cross_entropy(logits, targets)
        return loss
            
    def generate(self, tokenizer, context='', generate_len=10):
        buffer: list = tokenizer.encode(context)
        with torch.no_grad():
            for idx in np.arange(generate_len):
                if len(buffer) > self.context_len:
                    context = torch.Tensor(buffer[-self.context_len:]).to(self.device)
                else:
                    context = torch.zeros(self.context_len).to(self.device)
                    context[-len(buffer):] = torch.Tensor(buffer)
                context = torch.unsqueeze(context, dim=0)
                logits = self.forward(context.to(torch.long))
                probs = F.softmax(torch.squeeze(logits), dim=-1)
                # implement mask to shield rest of string
                new_token = torch.multinomial(probs, 1)[-1].item()
                buffer.append(new_token)
        print(buffer)
        print(tokenizer.decode(buffer))
                
                
                




if __name__ == "__main__":
    N = 2
    L = 3
    embed_dim = 4
    n_tokens = 5

    mask = torch.tril(torch.ones(L, L))
    print(mask)