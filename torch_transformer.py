import torch 
from torch import nn
from torch.nn import functional as F

char_to_idx = {}
idx_to_char = {}

mode = "generation"
dmodel = 65#dimension of the embedding
dk, dq, dv = 4, 4, 4 # for now, test numbers. 
num_heads = 16
num_layers = 4
block_size = 32
batch_size = 16
dropout_prob = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# create self-attention
class SelfAttention(nn.Module):
    def __init__(self, dmodel, dk, dq, dv):
        super.__init__()
        self.query = nn.Linear(dmodel, dq)
        self.key = nn.Linear(dmodel, dk)
        self.value = nn.Linear(dmodel, dv)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # x is of size (batch_size, block_size, dmodel)
        keys = self.key(x)
        query = self.query(x)
        vals = query @ keys.transpose(-2, -1) / (dmodel ** 0.5)
        vals = vals.masked_fill(torch.tril(vals), float('-inf'))
        vals = F.softmax(vals, dim=-1)
        vals = self.dropout(vals)
        out = vals @ self.value(x)
        return out
    
class MultiHeadSA(nn.Module):
    def __init__(self, dmodel, dk, dq, dv, num_heads):
        super.__init__()
        self.heads = nn.ModuleList([SelfAttention(dmodel, dk, dq, dv) for _ in range(num_heads)])
        self.linear = nn.Linear(dv*num_heads, dmodel)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.linear(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dmodel):
        super.__init__()
        global dropout_prob 
        self.net = nn.Sequential(
            nn.Linear(dmodel, dmodel),
            nn.ReLU(),
            nn.Linear(dmodel, dmodel),
            nn.Dropout(dropout_prob)
        )

    def forward(self, x):
        return self.net(x)
    
class TransformerBlock(nn.Module):
    def __init__(self):
        global dmodel, dk, dq, dv, num_heads
        super.__init__()
        self.mhsa = MultiHeadSA(dmodel, dk, dq, dv, num_heads)
        self.ff = FeedForward(dmodel)
        self.norm1 = nn.LayerNorm(dmodel)
        self.norm2 = nn.LayerNorm(dmodel)

    def forward(self, x):
        x = x + self.mhsa(self.norm1(x))   
        x = x + self.ff(self.norm2(x))
        return x
    
class Transformer(nn.Module):
    def __init__(self, num_layers):
        super.__init__()
        #learn some embeddings
        self.embed = nn.Embedding(len(char_to_idx), dmodel)
        self.pos_emb = nn.Embedding(block_size, dmodel)
        self.layers = nn.Sequential(*[TransformerBlock() for _ in range(num_layers)])
        self.linear = nn.Linear(dmodel, len(char_to_idx))
        self.norm = nn.LayerNorm(dmodel)
    
    def forward(self, x, y=None):
        # x is of size (batch_size, block_size)
        x = self.embed(x) + self.pos_emb(torch.arange(block_size))
        x = self.layers(x)
        x = self.linear(self.norm(x))
        if y is not None:
            logits = x.view(batch_size*block_size, dmodel)
            y = y.view(batch_size*block_size)
            loss = F.cross_entropy(logits, y)
            return logits, loss
        else:
            return logits, None
        
    
def transformer_generate_train(X, Y, model, max_iters=5000):
    #X and Y are already batched
    optimizer = nn.optimizers.Adam(model.parameters(), lr=0.001)

    #run iterations
    X = X.to(device)
    Y = Y.to(device)
    for _ in range(max_iters):
        logits, loss = model(X, Y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


