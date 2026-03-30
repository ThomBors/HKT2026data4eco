import numpy as np
import pandas as pd
import torch

from enviorment import Environment
from agent import Agent


# -----------------------------
# LOAD YOUR DATA
# -----------------------------
from data.dataloader import TimeSeriesDataset  # adjust name

df_train, df_weather_train, df_weather_test = TimeSeriesDataset.get_raw_data()

# merge production + weather
df_all = df_train[['Pcor']].join(df_weather_train, how='inner')

# -----------------------------
# TIME SPLIT
# -----------------------------
train_end = "2025-05-31"
val_end   = "2025-06-30"
test_start = "2025-12-31"

df_train_split = df_all.loc[:train_end]
df_val_split   = df_all.loc["2025-06-01":val_end]
df_test_split  = df_all.loc[test_start:]

# -----------------------------
# FEATURES
# -----------------------------
feature_cols = [c for c in df_all.columns if c != 'Pcor']

# -----------------------------
# ACTION BINS (from TRAIN only!)
# -----------------------------
delta = df_train_split['Pcor'].diff().fillna(0)
bins = np.linspace(delta.min(), delta.max(), 11)

# -----------------------------
# ENV + AGENT
# -----------------------------
print('ENV + AGENT')
env = Environment(df_train_split, feature_cols, bins)

state_dim = len(feature_cols)
action_dim = len(bins)

agent = Agent(state_dim, action_dim)

# -----------------------------
# TRAINING
# -----------------------------
episodes = 10

print('start training')

for ep in range(episodes):

    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)

        agent.store(state, action, reward, next_state, done)
        agent.train_step()

        state = next_state
        total_reward += reward

    agent.update_target()

    print(f"Episode {ep} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")


# -----------------------------
# VALIDATION (optional but useful)
# -----------------------------
val_env = Environment(df_val_split, feature_cols, bins)

state = val_env.reset()
done = False
val_reward = 0

while not done:
    action = agent.act(state)  # greedy-ish (epsilon low)
    next_state, reward, done = val_env.step(action)

    state = next_state
    val_reward += reward

print(f"\nValidation Reward: {val_reward:.2f}")


# -----------------------------
# TEST PREDICTION (KEY PART)
# -----------------------------
test_env = Environment(df_test_split, feature_cols, bins)

state = test_env.reset()
done = False

predictions = []
real_values = []

while not done:

    action = agent.act(state)

    # get current real production
    p_t = test_env.df.loc[test_env.t, 'Pcor']
    delta = bins[action]

    # predicted next production
    p_pred = p_t + delta

    # store prediction
    predictions.append(p_pred)

    # real value (if exists)
    if test_env.t + 1 < len(test_env.df):
        real = test_env.df.loc[test_env.t + 1, 'Pcor']
        real_values.append(real)

    state, reward, done = test_env.step(action)


# -----------------------------
# RESULTS
# -----------------------------
predictions = np.array(predictions)
real_values = np.array(real_values[:len(predictions)])

print("\nTest predictions (first 10):")
print(predictions[:10])

# simple metric
if len(real_values) > 0:
    mae = np.mean(np.abs(predictions - real_values))
    print(f"Test MAE: {mae:.4f}")