from core.base_module import BaseTrainableModule, BaseModule

class BaseEncoder(BaseTrainableModule):
    def forward(self, inputs: dict) -> dict:
        # expects inputs["state"]
        # returns {"latent": z, ...}
        raise NotImplementedError

class BaseWorldModel(BaseTrainableModule):
    def forward(self, inputs: dict) -> dict:
        # expects inputs["latent"] and inputs["action"]
        # returns {"next_latent": z_next, "reward": r_pred, ...}
        raise NotImplementedError

class BasePolicy(BaseTrainableModule):
    def forward(self, inputs: dict) -> dict:
        # expects inputs["latent"]
        # returns {"action_logits": logits, "value": V} or {"action": a, "value": V}
        raise NotImplementedError

class BasePlanner(BaseModule):
    def forward(self, inputs: dict) -> dict:
        # expects inputs["latent"], possibly world model
        # returns {"planned_action": a, "trajectory": sim_trajectory}
        raise NotImplementedError

class BaseCuriosity(BaseTrainableModule):
    def forward(self, inputs: dict) -> dict:
        # expects inputs including latent, action, next_latent
        # returns {"intrinsic_reward": r_int}
        raise NotImplementedError

class BaseSymbolicModule(BaseModule):
    def forward(self, inputs: dict) -> dict:
        # expects inputs["state"] or "latent"
        # returns {"constraints": mask_tensor, "program": program_repr}
        raise NotImplementedError
