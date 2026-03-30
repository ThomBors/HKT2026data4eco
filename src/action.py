import numpy as np

class Action:
    """
    Discrete action space: change in production (ΔPcor)
    """

    def __init__(self, bins):
        self.bins = bins
        self.n_actions = len(bins)

    def get_delta(self, action_idx):
        return self.bins[action_idx]

    def sample(self):
        return np.random.randint(self.n_actions)