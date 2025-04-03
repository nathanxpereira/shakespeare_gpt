import torch
import torch.nn as nn
import lightning as L

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"

        self.values  = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys    = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out  = nn.Linear(heads * self.head_dim, embed_size) # something to modify / parameterize?

    def forward(self, values, keys, query, mask=None):
        # usually value_len = key_len = query_len
        # B: Batch, S: Sequence Length, E: Embed Dim
        # values.shape = (B, S, E)
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        values  = values.reshape(N, value_len, self.heads, self.head_dim)
        keys    = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values  = self.values(values)
        keys    = self.keys(keys)
        queries = self.queries(queries)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-inf")) # -inf evaluates to zero after softmax
            
        attention = torch.softmax(energy / self.embed_size**0.5, dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)

        return self.fc_out(out)
    

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        return self.dropout(self.norm2(forward + x))
        
class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, heads, num_layers, max_length, src_vocab_size, dropout=0.1, device='cpu'):
        super(TransformerEncoder, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.max_length = max_length
        
        self.input_embedding    = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        N, seq_length = x.shape[:2]
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.input_embedding(x) + self.position_embedding(positions))
        
        for layer in self.layers:
            out = layer(out, out, out, mask)
            
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super(DecoderBlock,self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_size)


    def forward(self, value, key, x, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        out = self.dropout(self.norm(x+attention))
        out = self.transformer_block(value, key, out, src_mask)
        return out

class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, heads, num_layers, max_length, trg_vocab_size, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.output_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([DecoderBlock(embed_size, heads, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.output_activation = nn.Sigmoid()


    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape[:2]
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        out = self.dropout(self.output_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(enc_out, enc_out, out, src_mask, trg_mask)

        out = self.output_activation(self.fc_out(out))

        return out
    
class Transformer(nn.Module):
    def __init__(self, embed_size, heads, num_layers, max_length, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, dropout=0.1, device = 'cpu'):
        super(Transformer, self).__init__()

        self.encoder = TransformerEncoder(embed_size, heads, num_layers, max_length, src_vocab_size, dropout, device=device).to(device)
        self.decoder = TransformerDecoder(embed_size, heads, num_layers, max_length, trg_vocab_size, dropout).to(device)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def calculate_loss(self):

        loss = 0
        
    # TEMPORARY SRC MASK. ADJUST AS GO
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)   
        
        return out    

# Example usage:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], 
                      [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], 
                        [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)
    # Example parameters
    embed_size = 512
    heads      = 8
    num_layers = 6
    max_length = 100
    dropout    = 0

    batch_size = 16
    seq_length = 50
    
    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10

    # Create a Transformer encoder instance
    #encoder   = TransformerEncoder(embed_size, heads, num_layers, max_length)
    model = Transformer(embed_size=embed_size, 
                            heads=heads, 
                            num_layers=num_layers, 
                            max_length=max_length, 
                            src_vocab_size=src_vocab_size, 
                            trg_vocab_size=trg_vocab_size,
                            src_pad_idx=src_pad_idx,
                            trg_pad_idx=trg_pad_idx,
                            dropout=dropout,
                            device=device)

    # Forward pass through the encoder
    #encoded = encoder(x)
    with torch.no_grad():
        out = model(x, trg[:, :-1])
        print(out)

    # Output shape
    # print("Output shape:", encoded.shape)