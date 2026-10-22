"""Student network: depth image and body state to a velocity command."""

import numpy as np
import torch
from torch import nn

from task import ACTION_DIM, OBS_HEIGHT, OBS_WIDTH, STATE_DIM, V_MAX_M_S, YAW_RATE_MAX_RAD_S


class ApproachPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            features = self.conv(torch.zeros(1, 1, OBS_HEIGHT, OBS_WIDTH)).shape[1]
        self.head = nn.Sequential(
            nn.Linear(features + STATE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_DIM),
        )

    def forward(self, depth, state):
        return self.head(torch.cat((self.conv(depth), state), dim=-1))


def load_policy(path, device="cpu"):
    policy = ApproachPolicy()
    policy.load_state_dict(torch.load(path, map_location=device))
    policy.eval()
    return policy


def predict(policy, depth_obs, state, device="cpu"):
    """depth_obs is the normalised observation from task.downsample_depth."""
    with torch.no_grad():
        depth = torch.as_tensor(depth_obs, dtype=torch.float32, device=device).view(1, 1, OBS_HEIGHT, OBS_WIDTH)
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).view(1, STATE_DIM)
        action = policy(depth, state_tensor).cpu().numpy()[0]
    action = np.asarray(action, dtype=np.float32)
    horizontal = float(np.hypot(action[0], action[1]))
    if horizontal > V_MAX_M_S:
        action[0] *= V_MAX_M_S / horizontal
        action[1] *= V_MAX_M_S / horizontal
    action[2] = np.clip(action[2], -V_MAX_M_S, V_MAX_M_S)
    action[3] = np.clip(action[3], -YAW_RATE_MAX_RAD_S, YAW_RATE_MAX_RAD_S)
    return action
