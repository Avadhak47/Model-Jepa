import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from modules.interfaces import BasePolicy

class PPOPolicy(BasePolicy):
    """Proximal Policy Optimization Actor-Critic with Exploration Entropy tracking."""
    def __init__(self, config: dict):
        super().__init__(config)
        self.latent_dim = config.get("latent_dim", 128)
        self.action_dim = config.get("action_dim", 10) # 10 discrete actions
        self.continuous = config.get("continuous_actions", False)
        
        self.core = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh()
        )
        
        self.actor_mean = nn.Linear(256, self.action_dim)
        if self.continuous:
            self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))
            
        self.critic = nn.Linear(256, 1)
        self.to(self.device)

    def forward(self, inputs: dict) -> dict:
        z = inputs["latent"].to(self.device)
        features = self.core(z)
        
        value = self.critic(features)
        
        if self.continuous:
            action_mean = self.actor_mean(features)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        else:
            action_logits = self.actor_mean(features)
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        entropy = dist.entropy().mean()
        
        return {
            "action": action,
            "log_prob": log_prob,
            "value": value,
            "entropy": entropy,
            "action_dist": dist
        }

    def loss(self, inputs: dict, outputs: dict) -> dict:
        """PPO Surrogate Clipping loss + Critic + Entropy bonus."""
        old_log_probs = inputs["old_log_probs"].to(self.device)
        advantages = inputs["advantages"].to(self.device)
        returns = inputs["returns"].to(self.device)
        
        dist = outputs["action_dist"]
        action = inputs["taken_actions"].to(self.device)
        
        if self.continuous:
            curr_log_probs = dist.log_prob(action).sum(dim=-1)
        else:
            curr_log_probs = dist.log_prob(action)
            
        value = outputs["value"].squeeze(-1)
        entropy = dist.entropy().mean()
        
        ratios = torch.exp(curr_log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - 0.2, 1.0 + 0.2) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = F.mse_loss(value, returns)
        
        ent_coef = self.config.get("ent_coef", 0.01)
        loss = policy_loss + 0.5 * value_loss - ent_coef * entropy
        
        return {
            "loss": loss,
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy": entropy.detach()
        }

class DQNPolicy(BasePolicy):
    """Discrete Q-Learning policy."""
    def __init__(self, config: dict):
        super().__init__(config)
        self.latent_dim = config.get("latent_dim", 128)
        self.action_dim = config.get("action_dim", 10)
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)
        )
        self.to(self.device)

    def forward(self, inputs: dict) -> dict:
        z = inputs["latent"].to(self.device)
        q_values = self.net(z)
        action = torch.argmax(q_values, dim=-1)
        return {"q_values": q_values, "action": action}

    def loss(self, inputs: dict, outputs: dict) -> dict:
        q_values = outputs["q_values"]
        action = inputs["taken_actions"].to(self.device).long()
        target_q = inputs["target_q"].to(self.device)
        q_a = q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)
        loss = F.mse_loss(q_a, target_q)
        return {"loss": loss}

class DecisionTransformerPolicy(BasePolicy):
    """Offline sequence-conditioned return-to-go policy."""
    def __init__(self, config: dict):
        super().__init__(config)
        self.latent_dim = config.get("latent_dim", 128)
        self.action_dim = config.get("action_dim", 10)
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim + self.action_dim + 1, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)
        )
        self.to(self.device)

    def forward(self, inputs: dict) -> dict:
        z = inputs["latent"].to(self.device)
        a = inputs.get("prev_action", torch.zeros(z.shape[0], self.action_dim, device=self.device))
        rtg = inputs.get("return_to_go", torch.zeros(z.shape[0], 1, device=self.device))
        x = torch.cat([z, a, rtg], dim=-1)
        action_preds = self.net(x)
        return {"action_logits": action_preds, "action": action_preds.argmax(dim=-1)}
