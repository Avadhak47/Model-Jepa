import numpy as np
from envs.base_env import BaseEnvironment

class ARCEnvironment(BaseEnvironment):
    """Environment specific to Abstraction and Reasoning Corpus tasks."""
    def __init__(self, config: dict):
        super().__init__(**config)
        self.input_grid = np.zeros((10, 10))
        self.target_grid = np.ones((10, 10))
        self.state = self.input_grid.copy()
        self.action_space_size = config.get("action_dim", 10)

    def reset(self) -> dict:
        self.state = self.input_grid.copy()
        return {"state": self.state}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        action_val = int(np.argmax(action)) if np.size(action) > 1 else int(action)
        
        self.state = (self.state + action_val) % 10
        
        reward = self.compute_reward(self.state, self.target_grid)
        done = self.is_goal(self.state, self.target_grid)
        
        return {"state": self.state}, reward, done, {}

    def render(self):
        print("ARC Grid State:", self.state)

    def sample_action(self) -> np.ndarray:
        """Returns a random valid action via uniform categorical sampling."""
        action = np.zeros(self.action_space_size)
        idx = np.random.randint(0, self.action_space_size)
        action[idx] = 1.0
        return action

    def compute_reward(self, grid, target) -> float:
        if grid.shape != target.shape:
            return 0.0
        return float(np.sum(grid == target) / grid.size)

    def is_goal(self, grid, target) -> bool:
        if grid.shape != target.shape:
            return False
        return np.array_equal(grid, target)
