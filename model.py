from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class config:
    n_layer: int = 6
    n_embed: int = 512
    n_head: int = 8
    vocab_size: int = 8528
    block_size: int = 128
    dropout: int = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.mlp = nn.Sequential(
            nn.Linear(self.config.n_embed, 4 * self.config.n_embed),
            nn.GELU(),
            nn.Linear(4 * self.config.n_embed, self.config.n_embed)
        )

    def forward(self, x):
        return self.mlp(x)

class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        assert self.config.n_embed % self.config.n_head == 0

        self.hs = self.config.n_embed // self.config.n_head
        self.attn_dropout = nn.Dropout(self.config.dropout)
        self.resid_dropout = nn.Dropout(self.config.dropout)
        self.c_attn = nn.Linear(self.config.n_embed, 3 * self.config.n_embed)
        self.c_proj = nn.Linear(self.config.n_embed, self.config.n_embed)

        self.register_buffer("mask",
                             torch.tril(torch.ones(self.config.block_size, self.config.block_size, dtype = torch.bool).view(1, 1, self.config.block_size, self.config.block_size)))

    def forward(self, input):
        B, T, C = input.shape

        q, k, v = self.c_attn(input).split(self.config.n_embed, dim = -1)
        q_h = q.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)
        k_h = k.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)
        v_h = v.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)

        att = q_h @ k_h.transpose(-1, -2) / (self.hs ** 0.5)
        att = self.attn_dropout(att)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim = -1)

        y = att @ v_h
        y = y.transpose(1, 2).reshape(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.config = config
        self.ln1 = nn.LayerNorm(self.config.n_embed)
        self.ln2 = nn.LayerNorm(self.config.n_embed)
        self.mlp = MLP(self.config)
        self.c_attn = CausalAttention(self.config)

    def forward(self, x):
        x = x + self.c_attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Poetry_Cloud(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embed),
            wpe = nn.Embedding(self.config.block_size, self.config.n_embed),
            drop = nn.Dropout(self.config.dropout),
            h = nn.ModuleList([
                Block(self.config) for _ in range(self.config.n_layer)
            ]),
            ln_f = nn.LayerNorm(self.config.n_embed)
        ))

        self.lm_embed = nn.Linear(self.config.n_embed, self.config.vocab_size)
        self.transformer.wte.weight = self.lm_embed.weight

    def forward(self, x, target = None):
        device = x.device
        B, T = x.shape

        assert T <= self.config.block_size

        tok_em = self.transformer.wte(x)
        pos_em = self.transformer.wpe(torch.arange(0, T, device = device))

        x = self.transformer.drop(tok_em + pos_em)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if target is not None:
            logits = self.lm_embed(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index = -1)
        else:
            logits = self.lm_embed(x)
            loss = None

        return logits, loss

    @torch.no_grad
    def generate(self, idx, max_tokens = 100, temperature = 1.0, top_k = None, stop_id = None):
        for _ in range(max_tokens):
            idx_cond = idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim = -1)
            new_idx = torch.multinomial(probs, num_samples = 1)

            if stop_id is not None and (stop_id == new_idx).all():
                break

            idx = torch.cat((idx, new_idx), dim = 1)

        return idx

if __name__ == "__main__":
    x = torch.randint(0, 8528, (2, 128))
    model = Poetry_Cloud(config()).eval()
    logits, loss = model(x, x)
    print(loss.item())