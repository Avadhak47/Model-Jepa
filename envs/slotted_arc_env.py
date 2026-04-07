import numpy as np
import random
from envs.arc_env import ARCEnvironment
from envs.dsl import *

class SlottedARCEnvironment(ARCEnvironment):
    """
    ARC environment tailored for Slotted Architectures.
    Action space is a parameterized instruction rather than a flat integer.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.num_slots = config.get("num_slots", 16)
        
    def step_slotted(self, action_vector: tuple, slot_masks: np.ndarray) -> tuple[dict, float, bool, dict]:
        """
        Executes a parameterized DSL instruction on the grid.
        
        Args:
            action_vector: Tuple of (Action_Type, Target_Slot_ID, Parameter)
            slot_masks: Boolean array of shape [num_slots, max_grid, max_grid]
                        provided dynamically by the Agent's Slot Decoder during inference.
        """
        action_type, target_slot_idx, parameter = action_vector
        
        # Ensure safe slot indexing
        if target_slot_idx < 0 or target_slot_idx >= self.num_slots:
            # Invalid slot, return state unchanged
            self.step_count += 1
            reward = self.compute_reward(self.state, self.target_grid)
            done = self.is_goal(self.state, self.target_grid) or self.step_count >= self.max_steps
            return {"state": self.state.copy()}, reward, done, {"step": self.step_count}
            
        target_mask = slot_masks[target_slot_idx]
        
        if action_type == ACTION_TRANSLATE:
            # Parameter is direction: 0=Up, 1=Down, 2=Left, 3=Right
            dy, dx = 0, 0
            if parameter == 0: dy = -1
            elif parameter == 1: dy = 1
            elif parameter == 2: dx = -1
            elif parameter == 3: dx = 1
            self.state = translate(self.state, target_mask, dy, dx)
            
        elif action_type == ACTION_ROTATE:
            # Parameter is k (number of 90 deg rotations)
            self.state = rotate_object(self.state, target_mask, k=int(parameter))
            
        elif action_type == ACTION_FILL_COLOR:
            # Parameter is color 0-9
            color = int(parameter) % 10
            self.state = fill_color(self.state, target_mask, color)
            
        elif action_type == ACTION_BOUNDING_BOX:
            # Parameter is color 0-9
            color = int(parameter) % 10
            self.state = bounding_box(self.state, target_mask, color)
            
        elif action_type == ACTION_CONNECT:
            # Parameter is a tuple: (Target_Slot_B_ID, Color)
            if isinstance(parameter, tuple) and len(parameter) == 2:
                slot_b_idx, color = parameter
                if 0 <= slot_b_idx < self.num_slots:
                    mask_b = slot_masks[slot_b_idx]
                    self.state = connect(self.state, target_mask, mask_b, int(color) % 10)
                    
        self.step_count += 1
        reward = self.compute_reward(self.state, self.target_grid)
        done = self.is_goal(self.state, self.target_grid) or self.step_count >= self.max_steps
        
        return {"state": self.state.copy()}, reward, done, {"step": self.step_count}
