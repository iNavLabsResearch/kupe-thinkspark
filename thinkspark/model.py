"""ThinkSpark model — a tiny dual-encoder transformer with cross-attention fusion.

Why this architecture (not one flat encoder)
--------------------------------------------
The two inputs play different roles:

    INPUT   (the user's last utterance)   -> PRIMARY signal
    CONTEXT (running conversation state)   -> MODULATES the reaction

So we encode them separately, then let the INPUT tokens *cross-attend* into the
CONTEXT (input = query, context = key/value). The pooled representation is
therefore anchored on the input and only coloured by context — exactly the
"input is more important, context adjusts it" behaviour we want (e.g. the same
"okay" input yields an apologetic spark when context = 'user is scolding us').

Five classification heads share the fused trunk:
    intent (headline)  · language · register · emotion · filler_type

~1–3M params · byte vocab · runs comfortably on an M1 CPU/MPS.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .tokenizer import PAD_ID, VOCAB_SIZE


class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, ffn_mult, dropout, max_len, embed):
        super().__init__()
        self.embed = embed  # shared byte embedding
        self.pos = nn.Embedding(max_len, d_model)
        self.max_len = max_len
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * ffn_mult,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.scale = d_model ** 0.5

    def forward(self, ids, mask):
        b, t = ids.shape
        pos = torch.arange(t, device=ids.device).clamp_max(self.max_len - 1)
        x = self.embed(ids) * self.scale + self.pos(pos)[None]
        return self.enc(x, src_key_padding_mask=~mask)  # True = ignore(pad)


class ThinkSpark(nn.Module):
    def __init__(self, cfg_model, n_intent, n_lang, n_register, n_emotion, n_fillertype,
                 max_input_len, max_context_len):
        super().__init__()
        d = cfg_model.d_model
        self.embed = nn.Embedding(VOCAB_SIZE, d, padding_idx=PAD_ID)

        self.input_enc = Encoder(
            d, cfg_model.n_heads, cfg_model.input_layers, cfg_model.ffn_mult,
            cfg_model.dropout, max_input_len, self.embed,
        )
        self.context_enc = Encoder(
            d, cfg_model.n_heads, cfg_model.context_layers, cfg_model.ffn_mult,
            cfg_model.dropout, max_context_len, self.embed,
        )

        # Fusion: input tokens cross-attend into context (input = query).
        fuse_layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=cfg_model.n_heads, dim_feedforward=d * cfg_model.ffn_mult,
            dropout=cfg_model.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.fusion = nn.TransformerDecoder(fuse_layer, num_layers=cfg_model.fusion_layers)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(cfg_model.dropout)

        def head(n_out):
            return nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_out))

        self.head_intent = head(n_intent)
        self.head_lang = head(n_lang)
        self.head_register = head(n_register)
        self.head_emotion = head(n_emotion)
        self.head_fillertype = head(n_fillertype)

        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def _masked_mean(self, x, mask):
        m = mask.unsqueeze(-1).float()
        return (x * m).sum(1) / m.sum(1).clamp_min(1.0)

    def forward(self, batch: dict) -> dict:
        inp = self.input_enc(batch["input_ids"], batch["input_mask"])
        has_ctx = batch["context_mask"].any(dim=1, keepdim=True)  # (b,1)

        ctx = self.context_enc(batch["context_ids"], batch["context_mask"])
        # cross-attend: query=input, memory=context. Rows with empty context
        # would produce all-masked memory -> NaN; guard by feeding a permissive
        # key mask there and zeroing the contribution afterward.
        ctx_key_pad = ~batch["context_mask"]
        safe_key_pad = ctx_key_pad.clone()
        empty_rows = ~has_ctx.squeeze(1)
        safe_key_pad[empty_rows] = False  # let attention run, then null it out
        fused = self.fusion(
            tgt=inp, memory=ctx,
            tgt_key_padding_mask=~batch["input_mask"],
            memory_key_padding_mask=safe_key_pad,
        )
        # where there was no context, fall back to the pure input encoding
        gate = has_ctx.unsqueeze(-1).float()
        fused = gate * fused + (1.0 - gate) * inp

        pooled = self.norm(self._masked_mean(fused, batch["input_mask"]))
        pooled = self.drop(pooled)
        return {
            "intent": self.head_intent(pooled),
            "language": self.head_lang(pooled),
            "register": self.head_register(pooled),
            "emotion": self.head_emotion(pooled),
            "filler_type": self.head_fillertype(pooled),
        }


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
