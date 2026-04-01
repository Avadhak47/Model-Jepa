import torch
import torch.nn as nn

class BaseModule(nn.Module):
    """
    The foundational building block for all networks and logic engines.
    Operates strictly via the TensorDict protocol.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.device = config.get("device", "cpu")
        
    def forward(self, inputs: dict) -> dict:
        """Compute outputs from inputs. Must return a dict of tensors."""
        raise NotImplementedError("Subclasses must implement the forward method.")

    def to(self, device):
        """Override torch.nn.Module.to for convenience"""
        super().to(device)
        self.device = device
        return self

class BaseTrainableModule(BaseModule):
    """
    Extension of the base module for components that require gradient-based optimization.
    """
    def loss(self, inputs: dict, outputs: dict) -> dict:
        """
        Compute losses based on inputs and outputs.
        Returns a dict of {loss_name: scalar_tensor}, must include "loss" key for the main loss.
        """
        raise NotImplementedError("Subclasses of BaseTrainableModule must implement loss().")

    def update(self, loss_dict: dict):
        """
        Performs a backward pass on the main loss.
        """
        if "loss" in loss_dict:
            loss_dict["loss"].backward()
        else:
            raise KeyError("loss_dict must contain a 'loss' key for backpropagation.")
