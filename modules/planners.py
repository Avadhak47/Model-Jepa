import torch
import torch.nn as nn
from modules.interfaces import BasePlanner
import math

class MCTSPlanner(BasePlanner):
    """
    MuZero-style UCB MCTS using the differentiable world model as the simulator.

    Each call to forward() runs `num_simulations` rollout expansions from
    the root latent, accumulates visit counts via UCB1, and returns a
    normalised visit-count policy and a mean value estimate.

    Note: This is a latent-space MCTS — no image decoding is needed.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.num_simulations = config.get("num_simulations", 50)
        self.action_dim      = config.get("action_dim", 10)
        self.discount        = config.get("discount", 0.99)
        self.c1              = config.get("pb_c_init", 1.25)
        self.c2              = config.get("pb_c_base", 19652)
        self.latent_dim      = config.get("latent_dim", 128)

    # ── UCB score ────────────────────────────────────────────────────────
    def _ucb(self, q_values: torch.Tensor, visit_counts: torch.Tensor,
             total_visits: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        """Upper Confidence Bound for Trees (PUCT variant as in AlphaZero/MuZero)."""
        visit_sum  = total_visits.unsqueeze(-1).expand_as(visit_counts)
        exploration = prior * torch.sqrt(visit_sum) / (1.0 + visit_counts)
        exploration *= self.c1 + torch.log((visit_sum + self.c2 + 1.0) / self.c2)
        return q_values + exploration

    def forward(self, inputs: dict) -> dict:
        root_z       = inputs["latent"].to(self.device)          # [B, D]
        world_model  = inputs.get("world_model")
        policy_model = inputs.get("policy_model")
        B            = root_z.shape[0]

        # If no world model provided, return uniform safe defaults
        if world_model is None or policy_model is None:
            return {
                "mcts_policy": torch.ones(B, self.action_dim, device=self.device) / self.action_dim,
                "mcts_value":  torch.zeros(B, 1, device=self.device),
            }

        # ── Initialise tree statistics ─────────────────────────────────
        visit_counts = torch.zeros(B, self.action_dim, device=self.device)  # N(s, a)
        q_values     = torch.zeros(B, self.action_dim, device=self.device)  # Q(s, a)
        total_visits = torch.ones(B, device=self.device)                    # N(s)

        # Get prior policy from policy network
        with torch.no_grad():
            pol_out = policy_model({"latent": root_z})
            if "action_dist" in pol_out:
                prior = torch.softmax(pol_out["action_dist"].logits, dim=-1)
            else:
                # Fallback: extract logits from action
                prior = torch.ones(B, self.action_dim, device=self.device) / self.action_dim

        # ── Run simulations ────────────────────────────────────────────
        with torch.no_grad():
            for sim in range(self.num_simulations):
                # 1. SELECT action via UCB
                ucb_scores  = self._ucb(q_values, visit_counts, total_visits, prior)
                chosen_acts = ucb_scores.argmax(dim=-1)       # [B]

                # 2. EXPAND: build one-hot action, roll world model 1 step
                a_onehot = torch.zeros(B, self.action_dim, device=self.device)
                a_onehot.scatter_(1, chosen_acts.unsqueeze(-1), 1.0)

                wm_out   = world_model({
                    "latent":        root_z,
                    "action":        a_onehot,
                    "target_latent": root_z,           # dummy target
                    "target_reward": torch.zeros(B, device=self.device),
                })
                next_z   = wm_out["next_latent"].squeeze(1) if wm_out["next_latent"].dim() == 3 \
                           else wm_out["next_latent"]
                pred_r   = wm_out["predicted_reward"].squeeze(-1)
                if pred_r.dim() == 2:
                    pred_r = pred_r.squeeze(1)

                # 3. BACKUP: bootstrapped value from child policy
                child_pol = policy_model({"latent": next_z})
                child_val = child_pol["value"].squeeze(-1)    # [B]
                G         = pred_r + self.discount * child_val

                # 4. UPDATE statistics
                visit_counts.scatter_add_(1, chosen_acts.unsqueeze(-1), torch.ones(B, 1, device=self.device))
                q_values.scatter_(1, chosen_acts.unsqueeze(-1),
                                  ((q_values.gather(1, chosen_acts.unsqueeze(-1)) * visit_counts.gather(1, chosen_acts.unsqueeze(-1)) + G.unsqueeze(-1)) /
                                   (visit_counts.gather(1, chosen_acts.unsqueeze(-1)) + 1)))
                total_visits += 1.0

        # ── Extract results ─────────────────────────────────────────────
        # Policy: normalised visit distribution
        mcts_policy = visit_counts / visit_counts.sum(dim=-1, keepdim=True).clamp(min=1.0)
        # Value: expectation under visit distribution
        mcts_value  = (q_values * mcts_policy).sum(dim=-1, keepdim=True)

        return {"mcts_policy": mcts_policy, "mcts_value": mcts_value}


class CEMPlanner(BasePlanner):
    """
    Cross-Entropy Method planning in latent action space.
    Iteratively refines an action distribution by keeping elite samples.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.horizon     = config.get("cem_horizon", 5)
        self.n_samples   = config.get("cem_samples", 64)
        self.n_elite     = config.get("cem_elite", 16)
        self.n_iters     = config.get("cem_iters", 5)
        self.action_dim  = config.get("action_dim", 10)
        self.discount    = config.get("discount", 0.99)

    def forward(self, inputs: dict) -> dict:
        root_z      = inputs["latent"].to(self.device)     # [B, D]
        world_model = inputs.get("world_model")
        B           = root_z.shape[0]

        if world_model is None:
            return {
                "cem_action":  torch.zeros(B, self.action_dim, device=self.device),
                "cem_value":   torch.zeros(B, 1, device=self.device),
            }

        # Gaussian action distribution parameters [B, H, A]
        mu     = torch.zeros(B, self.horizon, self.action_dim, device=self.device)
        sigma  = torch.ones(B, self.horizon, self.action_dim, device=self.device)

        with torch.no_grad():
            for _ in range(self.n_iters):
                # Sample action sequences [B, S, H, A]
                noise    = torch.randn(B, self.n_samples, self.horizon, self.action_dim, device=self.device)
                act_seqs = mu.unsqueeze(1) + sigma.unsqueeze(1) * noise   # [B, S, H, A]

                # Evaluate each sample under the world model
                returns  = torch.zeros(B, self.n_samples, device=self.device)
                z_t      = root_z.unsqueeze(1).expand(-1, self.n_samples, -1)  # [B, S, D]
                z_t      = z_t.reshape(B * self.n_samples, -1)                  # [BS, D]

                for h in range(self.horizon):
                    a_h  = act_seqs[:, :, h, :].reshape(B * self.n_samples, -1)  # [BS, A]
                    a_oh = torch.softmax(a_h, dim=-1)
                    out  = world_model({
                        "latent":        z_t,
                        "action":        a_oh,
                        "target_latent": z_t,
                        "target_reward": torch.zeros(B * self.n_samples, device=self.device),
                    })
                    r    = out["predicted_reward"].squeeze(-1)
                    if r.dim() == 2: r = r.squeeze(1)
                    returns += (self.discount ** h) * r.reshape(B, self.n_samples)
                    z_t = out["next_latent"].squeeze(1) if out["next_latent"].dim() == 3 \
                          else out["next_latent"]

                # Refit distribution on elites
                topk_idx = returns.topk(self.n_elite, dim=-1).indices   # [B, n_elite]
                topk_idx_exp = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.horizon, self.action_dim)
                elites   = act_seqs.gather(1, topk_idx_exp)              # [B, n_elite, H, A]
                mu       = elites.mean(dim=1)
                sigma    = elites.std(dim=1).clamp(min=1e-3)

        best_action = torch.softmax(mu[:, 0, :], dim=-1)   # first-step action
        best_value  = returns.max(dim=-1, keepdim=True).values

        return {"cem_action": best_action, "cem_value": best_value}
