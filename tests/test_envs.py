import unittest
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG = {"latent_dim": 32, "action_dim": 8, "device": "cpu"}

class TestEnvironments(unittest.TestCase):
    """Ensure the simulated environment resets and steps correctly."""

    def setUp(self):
        from envs.arc_env import ARCEnvironment
        self.env = ARCEnvironment(CONFIG)

    def test_env_reset(self):
        obs = self.env.reset()
        self.assertIn("state", obs)
        self.assertIsInstance(obs["state"], np.ndarray)
        self.assertEqual(obs["state"].shape, (10, 10))

    def test_env_step(self):
        self.env.reset()
        action = self.env.sample_action()
        obs, reward, done, info = self.env.step(action)
        self.assertIn("state", obs)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)

    def test_sample_action_shape(self):
        action = self.env.sample_action()
        self.assertEqual(action.shape, (CONFIG["action_dim"],))
        # Only one-hot is set
        self.assertEqual(action.sum(), 1.0)

    def test_reward_range(self):
        self.env.reset()
        action = self.env.sample_action()
        _, reward, _, _ = self.env.step(action)
        self.assertGreaterEqual(reward, 0.0)
        self.assertLessEqual(reward, 1.0)

if __name__ == '__main__':
    unittest.main(verbosity=2)
