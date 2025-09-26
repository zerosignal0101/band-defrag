import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.band_defrag.mat_algorithm.model.mat_actions import (
    discrete_autoregreesive_act,
    discrete_parallel_act,
    continuous_autoregreesive_act,
    continuous_parallel_act,
)
from src.band_defrag.utils.mat_model_utils import check, init


# ---------------------------------------------------------------------------
#  IMPORTANT  –  Variable‑length agent support (padding + agent_mask)
# ---------------------------------------------------------------------------
# 1.  "max_agents" is fixed at network construction time.  All tensors coming
#     into the model are padded/trimmed to [B, max_agents, ...].
# 2.  A boolean / 0‑1 mask of shape [B, max_agents, 1] (or [B,max_agents]) is
#     carried through the whole forward pass.
# 3.  Wherever we do reductions (mean / sum) or attention, the mask is applied
#     so that padding positions contribute 0 and get no gradient.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# helpers --------------------------------------------------------------------
# ---------------------------------------------------------------------------


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain("relu")
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor):
    """
    x     : (B,N,D)
    mask  : (B,N)  1 = valid, 0 = padding
    return: (B,D)  平均后特征
    """
    while mask.dim() < x.dim():  # broadcast to (B,N,1)
        mask = mask.unsqueeze(-1)
    s = (x * mask).sum(1)  # (B,D)
    den = mask.sum(1).clamp(min=1e-6)  # (B,1)
    return s / den


# ---------------------------------------------------------------------------
# Self‑Attention that supports (1) causal masking (optional)  (2) key‑padding
# mask that zeroes out padded agents.
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, masked=False):
        super().__init__()
        assert n_embd % n_head == 0
        self.masked = masked  # "masked" == causal mask (decoder)
        self.n_head = n_head
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        self.proj = init_(nn.Linear(n_embd, n_embd))
        self.register_buffer("causal_cache", None, persistent=False)  # will build on the fly

    def forward(self, k_in, v_in, q_in, pad_mask=None):
        # pad_mask : (B, L) – 1 for valid agent, 0 for padding
        B, L, D = q_in.size()
        k = self.key(k_in).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        q = self.query(q_in).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        v = self.value(v_in).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, h, L, L)

        if self.masked:  # causal mask (for autoregressive decoding)
            if self.causal_cache is None or self.causal_cache.size(-1) != L:
                self.causal_cache = torch.tril(torch.ones(L, L, device=att.device)).view(1, 1, L, L)
            att = att.masked_fill(self.causal_cache == 0, -1e10)

        if pad_mask is not None:
            mask_q = (~pad_mask).unsqueeze(1).unsqueeze(2)  # (B,1,L,1)  Query 行
            mask_k = (~pad_mask).unsqueeze(1).unsqueeze(3)  # (B,1,1,L)  Key 列
            att = att.masked_fill(mask_q | mask_k, -1e10)
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, h, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        y = self.proj(y)
        return y


# ---------------------------------------------------------------------------
# Transformer blocks ---------------------------------------------------------
# ---------------------------------------------------------------------------
class EncodeBlock(nn.Module):
    def __init__(self, n_embd, n_head, masked=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, masked=masked)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 4 * n_embd, bias=True), activate=True),
            nn.GELU(),
            init_(nn.Linear(4 * n_embd, n_embd))
        )

    def forward(self, x, pad_mask=None):
        x = self.ln1(x + self.attn(x, x, x, pad_mask))
        x = self.ln2(x + self.mlp(x))
        if pad_mask is not None:
            x = x * pad_mask.unsqueeze(-1)
        return x


class DecodeBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.self_attn = SelfAttention(n_embd, n_head, masked=True)
        self.cross_attn = SelfAttention(n_embd, n_head, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 4 * n_embd, bias=True), activate=True),
            nn.GELU(),
            init_(nn.Linear(4 * n_embd, n_embd))
        )

    def forward(self, x, enc_rep, pad_mask=None):
        x = self.ln1(x + self.self_attn(x, x, x, pad_mask))
        x = self.ln2(x + self.cross_attn(enc_rep, enc_rep, x, pad_mask))
        x = self.ln3(x + self.mlp(x))
        return x


# ---------------------------------------------------------------------------
# Encoder --------------------------------------------------------------------
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, obs_dim, n_embd, n_head, n_block):
        super().__init__()
        self.token_emb = nn.Sequential(
            nn.LayerNorm(obs_dim),
            init_(nn.Linear(obs_dim, n_embd), activate=True),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([EncodeBlock(n_embd, n_head) for _ in range(n_block)])
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, obs, pad_mask=None):
        # obs: (B, maxN, obs_dim)
        x = self.token_emb(obs)
        for blk in self.blocks:
            x = blk(x, pad_mask)
        return self.ln(x)  # (B, maxN, D)


# ---------------------------------------------------------------------------
# Decoder / Actor ------------------------------------------------------------
# ---------------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, obs_dim, action_dim, n_embd, n_head, n_block,
                 action_type="Discrete"):
        super().__init__()
        self.action_type = action_type
        if action_type == "Discrete":
            self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True),
                                                nn.GELU())  # 离散动作多加一维“起始 <START> token”的占位符
        else:
            self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embd), activate=True), nn.GELU())
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.blocks = nn.ModuleList([DecodeBlock(n_embd, n_head) for _ in range(n_block)])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, action_dim)))

    def forward(self, actions, obs_rep, obs_raw, pad_mask=None):
        # one‑step decoding → produce logits for each agent independently
        action_embeddings = self.action_encoder(actions)
        x = self.ln(action_embeddings)  # (B,N,D)
        for blk in self.blocks:
            x = blk(x, obs_rep, pad_mask)
        logits = self.head(self.ln(x))  # (B,N,action_dim)
        return logits


# ---------------------------------------------------------------------------
# Multi‑Agent Transformer (variable agent version)
# ---------------------------------------------------------------------------
class MultiAgentTransformer(nn.Module):
    def __init__(
            self,
            obs_dim,
            action_dim,
            max_agents,
            n_block,
            n_embd,
            n_head,
            action_type="Discrete",
            device=torch.device("cpu"),
    ):
        super().__init__()
        self.max_agents = max_agents
        self.action_dim = action_dim
        self.action_type = action_type
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.encoder = Encoder(obs_dim, n_embd, n_head, n_block)
        self.decoder = Decoder(obs_dim, action_dim, n_embd, n_head, n_block, action_type)

        # twin Q
        def _make_q():
            return nn.ModuleDict({
                "obs_proj": nn.Sequential(
                    nn.LayerNorm(obs_dim),
                    init_(nn.Linear(obs_dim, n_embd), activate=True),
                    nn.GELU()
                ),
                "act_proj": nn.Sequential(
                    init_(nn.Linear(action_dim, n_embd), activate=True),
                    nn.GELU()
                ),
                "head": nn.Sequential(
                    init_(nn.Linear(n_embd, n_embd), activate=True),
                    nn.GELU(),
                    nn.LayerNorm(n_embd),
                    init_(nn.Linear(n_embd, 1))
                )
            })

        self.q1 = _make_q()
        self.q2 = _make_q()
        # fixed_alpha = 0.2
        # self.log_alpha = torch.tensor([math.log(fixed_alpha)], device=device)
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.to(device)

    # ---------------------------------------------------------------------
    # util
    # ---------------------------------------------------------------------
    @property
    def alpha(self):
        return self.log_alpha.exp()

    # ---------------------------------------------------------------------
    # core forward paths ---------------------------------------------------
    # ---------------------------------------------------------------------
    def forward(self, obs, action, agent_mask, ava=None):
        """Return logπ(a), Q1, Q2   (training‑time call)"""
        # obs       : (B,maxN,obs_dim)
        # action    : (B,maxN,1)  discrete index  or  (B,maxN,act_dim)
        # agent_mask: (B,maxN,1)
        pad_mask = agent_mask.squeeze(-1).bool()  # (B,maxN)
        obs_rep = self.encoder(obs, pad_mask)  # (B,N,D)

        B = obs.size(0)
        if self.action_type == "Discrete":
            act_idx = action.long()
            logp, _ = discrete_parallel_act(
                self.decoder, obs_rep, obs, act_idx, B,
                self.max_agents, self.action_dim, self.tpdv, ava, pad_mask
            )
        else:
            logp, _ = continuous_parallel_act(
                self.decoder, obs_rep, obs, action, B,
                self.max_agents, self.action_dim, self.tpdv
            )

        q1, q2 = self.get_q(obs, action, agent_mask)  # (B,1)
        return logp, q1, q2

    # @torch.no_grad()
    def get_actions(self, obs, agent_mask, ava=None, deterministic=False):
        pad_mask = agent_mask.squeeze(-1).bool()
        obs_rep = self.encoder(obs, pad_mask)
        B = obs.size(0)
        if self.action_type == "Discrete":
            act, logp = discrete_autoregreesive_act(
                self.decoder, obs_rep, obs, B,
                self.max_agents, self.action_dim, self.tpdv,
                ava, pad_mask, deterministic
            )
        else:
            act, logp = continuous_autoregreesive_act(
                self.decoder, obs_rep, obs, B,
                self.max_agents, self.action_dim, self.tpdv,
                deterministic
            )
        return act, logp

    # ------------------------------------------------------------------
    # critic ------------------------------------------------------------
    # ------------------------------------------------------------------
    def _forward_single_q(self, qnet, obs_emb, action_oh, mask):
        """
        obs        : [B, N, obs_dim]
        action_oh  : [B, N, action_dim]  (one-hot for discrete；raw for continuous)
        return     : [B, N]             (未做 mask)
        """
        # obs_emb = self.encoder(obs, mask)           # (B,N,D)
        act_emb = qnet["act_proj"](action_oh)  # (B,N,D)

        h = obs_emb + act_emb  # (B,N,D)
        g = _masked_mean(h, mask)  # (B,D)
        q = qnet["head"](g).squeeze(-1)  # (B,)
        return q

    def get_q(self, obs, action, agent_mask):
        """
        obs        : (B,N,obs_dim)
        action     : (B,N,1)   ―― discrete index  |  (B,N,act_dim) ―― continuous
        agent_mask : (B,N,1)   1 有效 / 0 padding
        ---------------------------------------------------------------
        返回 q1, q2  (B,N)  已乘 mask（padding 为 0）
        """
        obs = check(obs).to(**self.tpdv)
        agent_mask = agent_mask.squeeze(-1).bool()
        obs_emb = self.encoder(obs, agent_mask)
        agent_mask = check(agent_mask).to(**self.tpdv).squeeze(-1)
        if self.action_type == "Discrete":
            act_idx = action.squeeze(-1).long()  # 关键：先转 long
            act_oh = F.one_hot(act_idx, num_classes=self.action_dim).float()
        else:  # 连续动作直接用原值
            act_oh = action

        act_oh = check(act_oh).to(**self.tpdv)

        q1 = self._forward_single_q(self.q1, obs_emb, act_oh, agent_mask).unsqueeze(-1)
        q2 = self._forward_single_q(self.q2, obs_emb, act_oh, agent_mask).unsqueeze(-1)
        return q1, q2
