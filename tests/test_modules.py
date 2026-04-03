import unittest
import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG = {
    "latent_dim": 32, "action_dim": 8, "hidden_dim": 64,
    "num_layers": 2, "nhead": 4, "device": "cpu",
    "num_simulations": 5, "vocab_size": 20
}

class TestModules(unittest.TestCase):
    """Validates that all concrete modules adhere to the TensorDict protocol."""

    def _make_batch(self):
        return {
            "state":         torch.randn(2, 1, 8, 8),
            "latent":        torch.randn(2, CONFIG["latent_dim"]),
            "action":        torch.randn(2, CONFIG["action_dim"]),
            "target_latent": torch.randn(2, CONFIG["latent_dim"]),
            "target_reward": torch.zeros(2),
            "taken_actions": torch.zeros(2).long(),
            "target_q":      torch.zeros(2),
        }

    def test_mlp_encoder_forward(self):
        from modules.encoders import MLPEncoder
        enc = MLPEncoder(CONFIG)
        batch = self._make_batch()
        out = enc(batch)
        self.assertIn("latent", out)
        self.assertEqual(out["latent"].shape, (2, CONFIG["latent_dim"]))

    def test_cnn_encoder_forward(self):
        from modules.encoders import CNNEncoder
        enc = CNNEncoder(CONFIG)
        batch = self._make_batch()
        out = enc(batch)
        self.assertIn("latent", out)

    def test_mlp_dynamics_forward(self):
        from modules.world_models import MLPDynamicsModel
        wm = MLPDynamicsModel(CONFIG)
        batch = self._make_batch()
        out = wm(batch)
        self.assertIn("next_latent", out)
        self.assertEqual(out["next_latent"].shape, (2, CONFIG["latent_dim"]))

    def test_gaussian_dynamics_forward(self):
        from modules.world_models import GaussianDynamicsModel
        wm = GaussianDynamicsModel(CONFIG)
        batch = self._make_batch()
        out = wm(batch)
        self.assertIn("next_latent", out)
        self.assertIn("mean", out)
        self.assertIn("logvar", out)

    def test_transformer_world_model_forward(self):
        from modules.world_models import TransformerWorldModel
        wm = TransformerWorldModel(CONFIG)
        batch = self._make_batch()
        out = wm(batch)
        self.assertIn("next_latent", out)
        self.assertIn("attention_entropy", out)

    def test_ppo_policy_forward(self):
        from modules.policies import PPOPolicy
        pol = PPOPolicy(CONFIG)
        batch = self._make_batch()
        out = pol(batch)
        self.assertIn("action", out)
        self.assertIn("value", out)

    def test_dqn_policy_forward(self):
        from modules.policies import DQNPolicy
        pol = DQNPolicy(CONFIG)
        batch = self._make_batch()
        out = pol(batch)
        self.assertIn("q_values", out)
        self.assertIn("action", out)

    def test_rnd_curiosity_forward(self):
        from modules.curiosity import RNDCuriosity
        cur = RNDCuriosity(CONFIG)
        batch = self._make_batch()
        batch["next_latent"] = torch.randn(2, CONFIG["latent_dim"])
        out = cur(batch)
        self.assertIn("intrinsic_reward", out)

    def test_arc_dataset_sample(self):
        from arc_data.arc_dataset import ARCDataset
        ds = ARCDataset("./nonexistent_path")
        batch = ds.sample(4)
        self.assertEqual(batch["state"].shape, (4, 1, 30, 30))

    def test_tensor_dict_protocol(self):
        """All module outputs must be dicts."""
        from modules.encoders import MLPEncoder
        enc = MLPEncoder(CONFIG)
        out = enc(self._make_batch())
        self.assertIsInstance(out, dict)

    def test_backward_pass(self):
        """Gradients must flow through the World Model."""
        from modules.world_models import MLPDynamicsModel
        wm = MLPDynamicsModel(CONFIG)
        batch = self._make_batch()
        batch["latent"].requires_grad_(True)
        out = wm(batch)
        loss_dict = wm.loss(batch, out)
        loss_dict["loss"].backward()
        self.assertIsNotNone(batch["latent"].grad)

if __name__ == '__main__':
    unittest.main(verbosity=2)
