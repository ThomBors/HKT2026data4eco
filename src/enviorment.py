import numpy as np
from action import Action

class Environment:

    def __init__(self, df, feature_cols, bins):

        self.df = df.reset_index(drop=True)

        # --- clean data ---
        self.df[feature_cols] = self.df[feature_cols].ffill().bfill()
        self.df = self.df[self.df['Pcor'].notna()].reset_index(drop=True)

        self.feature_cols = feature_cols
        self.action_space = Action(bins)

        self.max_t = len(self.df) - 1
        self.t = 0

    def reset(self):
        self.t = 0
        return self._get_state()

    def _get_state(self):
        return self.df.loc[self.t, self.feature_cols].values.astype(np.float32)

    def step(self, action_idx):

        p_t = self.df.loc[self.t, 'Pcor']
        delta = self.action_space.get_delta(action_idx)

        # predicted next production
        p_pred = p_t + delta

        # real next production
        p_real = self.df.loc[self.t + 1, 'Pcor']

        # reward = prediction accuracy
        reward = -abs(p_real - p_pred)

        self.t += 1

        done = self.t >= self.max_t - 1

        next_state = self._get_state()

        return next_state, reward, done