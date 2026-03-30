import numpy as np
from enviorment import Environment   
from agent import Agent


def train_dqn(df, episodes=10):

    # -----------------------------
    # FEATURES
    # -----------------------------
    feature_cols = [c for c in df.columns if c != 'Pcor']

    # -----------------------------
    # ACTION SPACE
    # -----------------------------
    delta = df['Pcor'].diff().fillna(0)
    bins = np.linspace(delta.min(), delta.max(), 11)

    # -----------------------------
    # ENV + AGENT
    # -----------------------------
    env = Environment(df, feature_cols, bins)

    state_dim = len(feature_cols)
    action_dim = len(bins)

    agent = Agent(state_dim, action_dim)

    # -----------------------------
    # TRAIN LOOP
    # -----------------------------
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

    return agent, bins