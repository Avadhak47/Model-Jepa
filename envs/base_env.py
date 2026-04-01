class BaseEnvironment:
    """Gymnasium-like abstract environment for NS-ARC agent."""
    def __init__(self, **kwargs):
        self.config = kwargs

    def reset(self) -> dict:
        """Returns initial observation dict."""
        raise NotImplementedError

    def step(self, action) -> tuple[dict, float, bool, dict]:
        """Takes action, returns (obs_dict, extrinsic_reward, done, info_dict)."""
        raise NotImplementedError

    def render(self):
        pass

    def sample_action(self):
        """Return a random valid action."""
        raise NotImplementedError
