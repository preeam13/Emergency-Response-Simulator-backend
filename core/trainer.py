"""
OpenEnv Training Pipeline
GRPO (Group Relative Policy Optimization) via HuggingFace TRL
With curriculum learning, WandB logging, and checkpointing
"""

from __future__ import annotations

import os
import json
import time
import pickle
import random
import logging
from pathlib import Path
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("openenv.train")


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
@dataclass
class TrainConfig:
    # Environment
    num_ems: int = 2
    num_fire: int = 2
    num_police: int = 2
    num_dispatcher: int = 1
    grid_size: int = 32
    max_steps: int = 2000

    # Training
    total_episodes: int = 5000
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    batch_size: int = 64
    ppo_epochs: int = 4
    grad_clip: float = 0.5

    # GRPO specific
    group_size: int = 8       # G in GRPO: samples per group
    grpo_beta: float = 0.04   # KL regularization coefficient

    # Curriculum
    curriculum_thresholds: Tuple[float, ...] = (0.5, 0.65, 0.80)  # resolution rate gates

    # Checkpointing
    checkpoint_every: int = 100
    eval_every: int = 50
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # WandB
    use_wandb: bool = False
    wandb_project: str = "openenv-marl"
    run_name: str = "openenv-grpo-v1"

    # Hardware
    device: str = "auto"


# ─────────────────────────────────────────────
# Neural Network: Shared CNN + Actor-Critic
# ─────────────────────────────────────────────
class ConvEncoder(nn.Module):
    """Processes local observation patch (C x H x W)."""

    def __init__(self, in_channels: int, obs_size: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.LayerNorm([32, obs_size, obs_size]),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.LayerNorm([64, obs_size, obs_size]),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Flatten(),
        )
        conv_out = 64 * obs_size * obs_size
        self.proj = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x))


class MessageEncoder(nn.Module):
    """Encodes agent communication messages."""

    def __init__(self, msg_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(msg_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 64),
        )

    def forward(self, msgs: torch.Tensor) -> torch.Tensor:
        # msgs: (B, max_msgs, msg_dim) -> mean pool -> (B, 64)
        return self.net(msgs).mean(dim=1)


class ActorCriticAgent(nn.Module):
    """
    Per-agent Actor-Critic with:
    - CNN visual encoder for local obs
    - Message encoder for comms
    - Agent-type embedding
    - Shared trunk + separate actor/critic heads
    """

    def __init__(
        self,
        obs_channels: int,
        obs_size: int,
        action_dim: int,
        agent_type_dim: int = 4,
        msg_dim: int = 8,
    ):
        super().__init__()
        self.visual_enc = ConvEncoder(obs_channels, obs_size)
        self.msg_enc = MessageEncoder(msg_dim)
        self.type_emb = nn.Embedding(agent_type_dim, 16)

        trunk_in = 256 + 64 + 16
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

        # Initialize actor output layer close to uniform
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(
        self,
        obs: torch.Tensor,
        agent_type: torch.Tensor,
        msgs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        visual = self.visual_enc(obs)
        type_feat = self.type_emb(agent_type)

        if msgs is not None:
            msg_feat = self.msg_enc(msgs)
        else:
            msg_feat = torch.zeros(obs.shape[0], 64, device=obs.device)

        x = torch.cat([visual, msg_feat, type_feat], dim=-1)
        trunk_out = self.trunk(x)
        logits = self.actor(trunk_out)
        value = self.critic(trunk_out).squeeze(-1)
        return logits, value

    def get_action(
        self,
        obs: torch.Tensor,
        agent_type: torch.Tensor,
        msgs: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs, agent_type, msgs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


# ─────────────────────────────────────────────
# Rollout Buffer
# ─────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self, capacity: int, num_agents: int, obs_shape: tuple, device: str):
        self.capacity = capacity
        self.num_agents = num_agents
        self.device = device
        self.obs_shape = obs_shape
        self.clear()

    def clear(self):
        self.obs: List = []
        self.actions: List = []
        self.log_probs: List = []
        self.values: List = []
        self.rewards: List = []
        self.dones: List = []
        self.agent_types: List = []

    def add(self, obs, actions, log_probs, values, rewards, dones, agent_types):
        self.obs.append(obs)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.values.append(values)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.agent_types.append(agent_types)

    def compute_returns(self, gamma: float, gae_lambda: float, last_values: np.ndarray):
        T = len(self.rewards)
        n = self.num_agents
        returns = np.zeros((T, n), dtype=np.float32)
        advantages = np.zeros((T, n), dtype=np.float32)

        rewards = np.array(self.rewards)     # (T, n)
        values = np.array(self.values)       # (T, n)
        dones = np.array(self.dones)         # (T, n)

        gae = np.zeros(n)
        for t in reversed(range(T)):
            next_val = last_values if t == T - 1 else values[t + 1]
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_val * mask - values[t]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        return returns, advantages

    def get_batches(self, batch_size: int, returns, advantages):
        T = len(self.obs)
        n = self.num_agents
        indices = np.arange(T * n)
        np.random.shuffle(indices)

        obs_arr = np.array(self.obs).reshape(T * n, *self.obs_shape)
        act_arr = np.array(self.actions).reshape(T * n)
        lp_arr = np.array(self.log_probs).reshape(T * n)
        ret_arr = returns.reshape(T * n)
        adv_arr = advantages.reshape(T * n)
        typ_arr = np.array(self.agent_types).reshape(T * n)

        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            yield (
                torch.FloatTensor(obs_arr[idx]).to(self.device),
                torch.LongTensor(act_arr[idx]).to(self.device),
                torch.FloatTensor(lp_arr[idx]).to(self.device),
                torch.FloatTensor(ret_arr[idx]).to(self.device),
                torch.FloatTensor(adv_arr[idx]).to(self.device),
                torch.LongTensor(typ_arr[idx]).to(self.device),
            )


# ─────────────────────────────────────────────
# GRPO Loss
# ─────────────────────────────────────────────
def grpo_loss(
    logits: torch.Tensor,
    old_log_probs: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    clip_eps: float = 0.2,
    value_coeff: float = 0.5,
    entropy_coeff: float = 0.01,
    beta: float = 0.04,
) -> Tuple[torch.Tensor, Dict]:
    dist = Categorical(logits=logits)
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    # Group normalization of advantages (GRPO key innovation)
    adv_mean = advantages.mean()
    adv_std = advantages.std() + 1e-8
    normalized_adv = (advantages - adv_mean) / adv_std

    # PPO clipped surrogate
    ratio = torch.exp(new_log_probs - old_log_probs.detach())
    surr1 = ratio * normalized_adv
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * normalized_adv
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss
    value_loss = F.mse_loss(values, returns)

    # KL divergence regularization (GRPO beta term)
    kl = (old_log_probs.detach() - new_log_probs).mean()
    kl_penalty = beta * kl

    total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy + kl_penalty

    stats = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "kl": kl.item(),
        "total_loss": total_loss.item(),
        "ratio_mean": ratio.mean().item(),
    }
    return total_loss, stats


# ─────────────────────────────────────────────
# Curriculum Manager
# ─────────────────────────────────────────────
class CurriculumManager:
    def __init__(self, thresholds: Tuple[float, ...]):
        self.thresholds = thresholds
        self.level = 1
        self.recent_rates: deque = deque(maxlen=20)

    def update(self, resolution_rate: float) -> bool:
        self.recent_rates.append(resolution_rate)
        if len(self.recent_rates) < 10:
            return False

        avg = sum(self.recent_rates) / len(self.recent_rates)
        if self.level <= len(self.thresholds) and avg >= self.thresholds[self.level - 1]:
            self.level += 1
            log.info(f"Curriculum advanced to level {self.level} (avg rate={avg:.2f})")
            return True
        return False


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────
class OpenEnvTrainer:
    def __init__(self, config: TrainConfig):
        self.config = config

        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device

        log.info(f"Using device: {self.device}")

        # Import here to avoid circular
        from core.environment import OpenEnv, AgentType
        self.EnvClass = OpenEnv
        self.AgentType = AgentType

        # Obs / action sizes
        self.obs_size = 2 * 5 + 1  # OBS_RADIUS * 2 + 1 = 11
        self.obs_channels = 8
        self.action_dim = config.grid_size ** 2 + 1
        self.num_agents = (
            config.num_ems + config.num_fire + config.num_police + config.num_dispatcher
        )

        # Shared policy network (parameter sharing across agent types)
        self.policy = ActorCriticAgent(
            obs_channels=self.obs_channels,
            obs_size=self.obs_size,
            action_dim=self.action_dim,
            agent_type_dim=4,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.total_episodes, eta_min=1e-5
        )

        # Curriculum
        self.curriculum = CurriculumManager(config.curriculum_thresholds)

        # Logging
        self.training_log: List[Dict] = []
        self.episode_returns: List[float] = []
        self.eval_results: List[Dict] = []

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        # WandB
        self.wandb = None
        if config.use_wandb:
            try:
                import wandb
                self.wandb = wandb
                wandb.init(project=config.wandb_project, name=config.run_name, config=asdict(config))
                log.info("WandB initialized")
            except ImportError:
                log.warning("WandB not installed; running without logging")

    def _build_env(self, curriculum_level: int = 1):
        return self.EnvClass(
            num_ems=self.config.num_ems,
            num_fire=self.config.num_fire,
            num_police=self.config.num_police,
            num_dispatcher=self.config.num_dispatcher,
            grid_size=self.config.grid_size,
            seed=random.randint(0, 10000),
            curriculum_level=curriculum_level,
        )

    def _obs_to_tensor(self, obs_dict: Dict[int, np.ndarray]) -> torch.Tensor:
        """Stack all agent observations into (N, C, H, W) tensor."""
        obs_list = [obs_dict[i] for i in sorted(obs_dict.keys())]
        return torch.FloatTensor(np.stack(obs_list)).to(self.device)

    def _agent_types_tensor(self) -> torch.Tensor:
        """Returns agent type indices (N,)."""
        env = self._build_env()
        types = [a.type.value for a in env.agents]
        return torch.LongTensor(types).to(self.device)

    def collect_rollout(self, env, n_steps: int = 512) -> Tuple[RolloutBuffer, Dict]:
        buffer = RolloutBuffer(
            capacity=n_steps,
            num_agents=self.num_agents,
            obs_shape=(self.obs_channels, self.obs_size, self.obs_size),
            device=self.device,
        )

        obs_dict, _ = env.reset()
        agent_types = torch.LongTensor([a.type.value for a in env.agents]).to(self.device)

        episode_reward = 0.0
        for _ in range(n_steps):
            obs_t = self._obs_to_tensor(obs_dict)  # (N, C, H, W)

            with torch.no_grad():
                logits, values = self.policy(obs_t, agent_types)
                dist = Categorical(logits=logits)
                actions_t = dist.sample()
                log_probs_t = dist.log_prob(actions_t)

            actions = actions_t.cpu().numpy()
            log_probs = log_probs_t.cpu().numpy()
            vals = values.cpu().numpy()

            actions_dict = {a.id: int(actions[i]) for i, a in enumerate(env.agents)}
            obs_dict, rewards_dict, terminated, truncated, info = env.step(actions_dict)

            rewards_arr = np.array([rewards_dict.get(i, 0.0) for i in range(self.num_agents)])
            dones_arr = np.array([terminated.get(i, False) or truncated.get(i, False)
                                   for i in range(self.num_agents)], dtype=np.float32)

            buffer.add(
                obs=obs_t.cpu().numpy(),
                actions=actions,
                log_probs=log_probs,
                values=vals,
                rewards=rewards_arr,
                dones=dones_arr,
                agent_types=agent_types.cpu().numpy(),
            )
            episode_reward += rewards_arr.sum()

            if all(truncated.values()):
                break

        # Last values for GAE
        obs_t = self._obs_to_tensor(obs_dict)
        with torch.no_grad():
            _, last_values = self.policy(obs_t, agent_types)
        last_vals = last_values.cpu().numpy()

        returns, advantages = buffer.compute_returns(
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            last_values=last_vals,
        )
        summary = {
            "episode_reward": episode_reward,
            "resolution_rate": info.get("resolution_rate", 0),
            "lives_saved": info.get("total_lives_saved", 0),
            "lives_lost": info.get("total_lives_lost", 0),
            "cascade_count": info.get("cascade_count", 0),
            "avg_response_time": info.get("avg_response_time", 0),
        }
        return buffer, returns, advantages, summary

    def update_policy(self, buffer: RolloutBuffer, returns, advantages) -> Dict:
        all_stats = []

        for _ in range(self.config.ppo_epochs):
            for batch in buffer.get_batches(self.config.batch_size, returns, advantages):
                obs_b, act_b, old_lp_b, ret_b, adv_b, typ_b = batch

                logits, values = self.policy(obs_b, typ_b)

                loss, stats = grpo_loss(
                    logits=logits,
                    old_log_probs=old_lp_b,
                    actions=act_b,
                    advantages=adv_b,
                    returns=ret_b,
                    values=values,
                    clip_eps=self.config.clip_eps,
                    value_coeff=self.config.value_coeff,
                    entropy_coeff=self.config.entropy_coeff,
                    beta=self.config.grpo_beta,
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
                self.optimizer.step()
                all_stats.append(stats)

        self.scheduler.step()

        # Aggregate stats
        agg = {k: np.mean([s[k] for s in all_stats]) for k in all_stats[0]}
        return agg

    def evaluate(self, n_episodes: int = 5) -> Dict:
        self.policy.eval()
        results = []

        for _ in range(n_episodes):
            env = self._build_env(self.curriculum.level)
            obs_dict, _ = env.reset()
            agent_types = torch.LongTensor([a.type.value for a in env.agents]).to(self.device)

            ep_reward = 0.0
            for _ in range(self.config.max_steps):
                obs_t = self._obs_to_tensor(obs_dict)
                with torch.no_grad():
                    actions_t, _, _ = self.policy.get_action(obs_t, agent_types, deterministic=True)
                actions_dict = {a.id: int(actions_t[i]) for i, a in enumerate(env.agents)}
                obs_dict, rewards_dict, _, truncated, info = env.step(actions_dict)
                ep_reward += sum(rewards_dict.values())
                if all(truncated.values()):
                    break

            metrics = env.get_metrics_summary()
            metrics["episode_reward"] = ep_reward
            results.append(metrics)

        self.policy.train()
        agg = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
        return agg

    def save_checkpoint(self, episode: int, eval_metrics: Dict):
        path = Path(self.config.checkpoint_dir) / f"checkpoint_ep{episode:05d}.pt"
        torch.save({
            "episode": episode,
            "policy_state": self.policy.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "curriculum_level": self.curriculum.level,
            "eval_metrics": eval_metrics,
            "config": asdict(self.config),
        }, path)
        log.info(f"Checkpoint saved: {path}")
        return str(path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.curriculum.level = ckpt.get("curriculum_level", 1)
        log.info(f"Loaded checkpoint from {path} (episode {ckpt['episode']})")
        return ckpt["episode"]

    def train(self, resume_from: Optional[str] = None) -> List[Dict]:
        start_ep = 0
        if resume_from:
            start_ep = self.load_checkpoint(resume_from)

        log.info(f"Training for {self.config.total_episodes} episodes")
        best_eval_reward = float("-inf")

        for episode in range(start_ep, self.config.total_episodes):
            env = self._build_env(self.curriculum.level)

            # Collect rollout (512 steps per episode)
            buffer, returns, advantages, ep_summary = self.collect_rollout(env, n_steps=512)

            # Update policy
            update_stats = self.update_policy(buffer, returns, advantages)

            # Curriculum update
            self.curriculum.update(ep_summary["resolution_rate"])

            # Logging
            log_entry = {
                "episode": episode,
                "curriculum_level": self.curriculum.level,
                **ep_summary,
                **update_stats,
            }
            self.training_log.append(log_entry)
            self.episode_returns.append(ep_summary["episode_reward"])

            if episode % 10 == 0:
                recent_reward = np.mean(self.episode_returns[-10:])
                log.info(
                    f"Ep {episode:4d} | Level {self.curriculum.level} | "
                    f"Reward {recent_reward:8.2f} | "
                    f"ResRate {ep_summary['resolution_rate']:.2f} | "
                    f"Lives+ {ep_summary['lives_saved']:3d} | "
                    f"Loss {update_stats['total_loss']:.4f}"
                )

            if self.wandb and episode % 5 == 0:
                self.wandb.log(log_entry)

            # Evaluation
            if episode % self.config.eval_every == 0:
                eval_metrics = self.evaluate(n_episodes=3)
                self.eval_results.append({"episode": episode, **eval_metrics})
                log.info(
                    f"EVAL ep {episode:4d} | "
                    f"Reward {eval_metrics['episode_reward']:.2f} | "
                    f"SurvivalRate {eval_metrics['survival_rate']:.2f} | "
                    f"ResRate {eval_metrics['resolution_rate']:.2f}"
                )
                if self.wandb:
                    self.wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()})

                # Save best model
                if eval_metrics["episode_reward"] > best_eval_reward:
                    best_eval_reward = eval_metrics["episode_reward"]
                    self.save_checkpoint(episode, eval_metrics)

            # Regular checkpoint
            if episode % self.config.checkpoint_every == 0:
                self.save_checkpoint(episode, {})

        # Save training log
        log_path = Path(self.config.log_dir) / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(self.training_log, f, indent=2)
        log.info(f"Training complete. Log saved to {log_path}")

        if self.wandb:
            self.wandb.finish()

        return self.training_log


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def main():
    config = TrainConfig(
        total_episodes=2000,
        use_wandb=False,
        curriculum_thresholds=(0.45, 0.60, 0.75),
    )
    trainer = OpenEnvTrainer(config)
    logs = trainer.train()
    print(f"\nTraining complete. {len(logs)} episodes logged.")


if __name__ == "__main__":
    main()
