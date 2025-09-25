import torch
from torch.distributions import Categorical, Normal
from torch.nn import functional as F
from typing import Union


from typing import Union
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

# -----------------------------------------------------------------
# Safe (non-in-place) autoregressive action sampler
# -----------------------------------------------------------------
def discrete_autoregreesive_act(
    decoder,
    obs_rep, obs,
    batch_size, n_agent, action_dim, tpdv,
    available_actions: Union[torch.Tensor, None] = None,
    pad_mask:      Union[torch.Tensor, None] = None,
    deterministic: bool = False,
):
    """
    返回:
        actions : (B, N, 1)
        log_pi  : (B, N, 1)
    所有对带梯度张量的操作都 **非原地**，彻底避免 version mismatch。
    """
    actions, logps = [], []

    # decoder 第一个输入 token 是 <START>
    prev = torch.zeros(batch_size, 1, action_dim + 1, **tpdv)
    prev[:, 0, 0] = 1.0                         # (B,1,A+1)

    for t in range(n_agent):
        # 只喂到第 t 个 token → decoder 自带 causal mask
        logit_t = decoder(
            prev,
            obs_rep[:, :t + 1],
            obs[:, :t + 1],
            pad_mask[:, :t + 1] if pad_mask is not None else None
        )[:, -1, :]  # ← 不再切掉第 0 列

        if available_actions is not None:
            logit_t = logit_t.masked_fill(available_actions[:, t] == 0, -1e10)

        dist = Categorical(logits=logit_t)
        act  = dist.probs.argmax(-1) if deterministic else dist.sample()

        actions.append(act)
        logps.append(dist.log_prob(act))

        # 生成下一个 decoder 输入，不改 prev，而是拼接新 token
        next_tok = F.one_hot(act, num_classes=action_dim).float()       # (B,A)
        next_tok = torch.cat(
            [torch.zeros(batch_size, 1, device=next_tok.device), next_tok], dim=-1
        ).unsqueeze(1)                                                  # (B,1,A+1)
        prev = torch.cat([prev, next_tok], dim=1)                       # (B,t+2,A+1)

    actions = torch.stack(actions, dim=1).unsqueeze(-1)  # (B,N,1)
    log_pi  = torch.stack(logps,   dim=1).unsqueeze(-1)  # (B,N,1)

    if pad_mask is not None:
        pad = pad_mask.unsqueeze(-1)
        actions = actions * pad
        log_pi  = log_pi  * pad

    return actions, log_pi

def discrete_parallel_act(decoder,
                          obs_rep, obs, action,
                          batch_size, n_agent, action_dim, tpdv,
                          available_actions=None,
                          pad_mask: Union[torch.Tensor, None] = None):
    """
    训练时：一次性算整批给定动作的 log-π 与熵
    ---------------------------------------------------------------
    pad_mask : (B, N) 1→有效；0→padding
    """
    one_hot_action = F.one_hot(action.squeeze(-1), num_classes=action_dim)  # (B,N,A)

    shifted_action = torch.zeros(batch_size, n_agent, action_dim + 1, **tpdv)
    shifted_action[:, 0, 0] = 1.
    shifted_action[:, 1:, 1:] = one_hot_action[:, :-1, :]

    logit = decoder(shifted_action, obs_rep, obs, pad_mask)                          # (B,N,A)

    if available_actions is not None:
        logit[available_actions == 0] = -1e10
    if pad_mask is not None:
        # 让形状变成 (B, N, A) 与 logit 对齐
        pad_exp = pad_mask.unsqueeze(-1).expand(-1, -1, action_dim)
        logit = logit.masked_fill(~pad_exp, -1e10)  # 或 logit[~pad_exp] = -1e10

    dist = Categorical(logits=logit)
    act_log = dist.log_prob(action.squeeze(-1)).unsqueeze(-1)             # (B,N,1)
    entropy = dist.entropy().unsqueeze(-1)                                # (B,N,1)

    if pad_mask is not None:                                               # 使 padding 位贡献为 0
        act_log = act_log * pad_mask.unsqueeze(-1)
        entropy = entropy * pad_mask.unsqueeze(-1)

    return act_log, entropy


def continuous_autoregreesive_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                  deterministic=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    output_action = torch.zeros((batch_size, n_agent, action_dim), dtype=torch.float32)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    for i in range(n_agent):
        act_mean = decoder(shifted_action, obs_rep, obs)[:, i, :]
        action_std = torch.sigmoid(decoder.log_std) * 0.5

        # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
        # distri = Normal(act_mean, log_std.exp())
        distri = Normal(act_mean, action_std)
        action = act_mean if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action
        output_action_log[:, i, :] = action_log
        if i + 1 < n_agent:
            shifted_action[:, i + 1, :] = action

        # print("act_mean: ", act_mean)
        # print("action: ", action)

    return output_action, output_action_log


def continuous_parallel_act(decoder, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    shifted_action[:, 1:, :] = action[:, :-1, :]

    act_mean = decoder(shifted_action, obs_rep, obs)
    action_std = torch.sigmoid(decoder.log_std) * 0.5
    distri = Normal(act_mean, action_std)

    # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
    # distri = Normal(act_mean, log_std.exp())

    action_log = distri.log_prob(action)
    entropy = distri.entropy()
    return action_log, entropy
