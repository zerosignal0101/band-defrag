import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from mat_vat_agents.utils.util import get_gard_norm, huber_loss, mse_loss
from mat_vat_agents.utils.valuenorm import ValueNorm
from mat_vat_agents.algorithms.utils.util import check
from tensorboardX import SummaryWriter
from pathlib import Path
import os
import math

def _grad_norm(opt):
    total = 0.0
    for p in opt.param_groups[0]["params"]:
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5

class MATTrainer:
    """
    Trainer class for MAT to update policies.
    1. 从 Buffer 里取出一个 epoch 里所有 mini-batch；
    2. 按 PPO/MAPPO 公式计算损失；
    3. 调用 TransformerPolicy.evaluate_actions 前向，再 backward + Adam.step() 更新 Transformer 参数。
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.batch_size = args.batch_size
        # self.target_entropy = -float(policy.act_dim)
        self.target_entropy = -math.log(policy.act_dim)
        # === 调试相关 ===
        self.run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / args.env_name / args.algorithm_name / args.experiment_name
        self.debug_writer = SummaryWriter(str(self.run_dir / "critic_debug"))
        self._debug_step = 0  # 单独的计数器


    def train(self, buffer, num_trains=1):
        info = dict(actor_loss=0, critic_loss=0, alpha_loss=0, alpha=self.policy.get_alpha().item())

        for _ in range(num_trains):
            batch = buffer.sample(self.batch_size)
            obs = batch["obs"].to(self.device)
            actions = batch["actions"].to(self.device)
            mask = batch["masks"].to(self.device)
            available_actions = batch.get("ava", None)
            if available_actions is not None:
                available_actions = available_actions.to(self.device)
            reward = batch["rewards"].to(self.device)        # shape (B,1)
            # -------------  Critic  -------------
            with torch.no_grad():
                target_q = reward                            # 单步 → 目标就是 r
            _, q1_pred, q2_pred = self.policy.evaluate_actions(obs, mask, actions, available_actions)
            td_err1 = (q1_pred - target_q)
            td_err2 = (q2_pred - target_q)
            # critic_loss = (td_err1.pow(2).mean() + td_err2.pow(2).mean())
            critic_loss = F.smooth_l1_loss(q1_pred, target_q) + \
                          F.smooth_l1_loss(q2_pred, target_q)

            abs_err1 = td_err1.abs()
            abs_err2 = td_err2.abs()

            self.policy.q1_opt.zero_grad()
            self.policy.q2_opt.zero_grad()
            critic_loss.backward()
            self.policy.q1_opt.step()
            self.policy.q2_opt.step()

            # -------------  Actor  -------------
            new_actions, log_pi = self.policy.get_actions(obs, mask, available_actions, deterministic=False)
            q1_pi, q2_pi = self.policy.transformer.get_q(obs, new_actions, mask)
            q_min = torch.min(q1_pi, q2_pi)

            pad = mask.squeeze(-1).float()  # (B,N)
            valid_cnt = pad.sum(1, keepdim=True).clamp(min=1)
            log_pi_mean = (log_pi.squeeze(-1) * pad).sum(1, keepdim=True) / valid_cnt

            alpha = self.policy.get_alpha()
            actor_loss = (alpha.detach()*log_pi_mean - q_min).mean()
            self.policy.actor_opt.zero_grad()
            actor_loss.backward()
            # print("actor grad norm =", _grad_norm(self.policy.actor_opt))

            self.policy.actor_opt.step()

            # -------------  α  ------------------
            alpha_loss = -(self.policy.transformer.log_alpha * (log_pi_mean + self.target_entropy).detach()).mean()
            self.policy.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.policy.alpha_opt.step()

            # --- TensorBoard logging ---
            if self._debug_step % 10 == 0:
                step = self._debug_step
                wrt = self.debug_writer

                # Q 值
                wrt.add_scalar("q1_mean", q1_pred.mean(), step)
                wrt.add_scalar("q2_mean", q2_pred.mean(), step)
                wrt.add_scalar("target_q_mean", target_q.mean(), step)
                wrt.add_scalar("target_q_std", target_q.std(), step)

                # 误差统计（绝对误差 + RMS）
                wrt.add_scalar("abs_err1_mean", abs_err1.mean(), step)
                wrt.add_scalar("abs_err2_mean", abs_err2.mean(), step)
                wrt.add_scalar("td_err1_rms", abs_err1.pow(2).mean().sqrt(), step)
                wrt.add_scalar("td_err2_rms", abs_err2.pow(2).mean().sqrt(), step)

                wrt.add_histogram("abs_err1_hist", abs_err1, step)
                wrt.add_histogram("abs_err2_hist", abs_err2, step)

                # 梯度
                wrt.add_scalar("grad_norm_q1", _grad_norm(self.policy.q1_opt), step)
                wrt.add_scalar("grad_norm_q2", _grad_norm(self.policy.q2_opt), step)

                # 在 policy.get_actions 之后
                wrt.add_scalar("actor/log_pi_mean", log_pi_mean.mean(), step)
                wrt.add_scalar("actor/q_min", q_min.mean(), step)
                wrt.add_scalar("actor/grad_norm", _grad_norm(self.policy.actor_opt), step)

            self._debug_step += 1

            # -------------  log ----------------
            info["actor_loss"] += actor_loss.item()
            info["critic_loss"] += critic_loss.item()
            info["alpha_loss"] += alpha_loss.item()
            info["alpha"] = self.policy.get_alpha().item()

        for k in info:
            info[k] /= num_trains
        return info

    # rollout阶段，eval()关闭dropout、layer-norm训练态
    # training阶段，train()开启梯度
    def prep_training(self):
        self.policy.train()

    def prep_rollout(self):
        self.policy.eval()
