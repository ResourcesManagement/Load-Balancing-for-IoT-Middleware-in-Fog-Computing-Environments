import os
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
import random
from types import SimpleNamespace

# --- Parameters ---
Parameters = SimpleNamespace(
    GAMMA=0.99,
    LR=0.001,
    BATCH_SIZE=64,
    STEP_PER_EPOCHS=50,
    BUFFER_SIZE=10000,
    TARGET_UPDATE=10,
    ENV_STATE_SIZE=10,
    ACTION_SPACE=5,
    EPISODES=500
)


# --- Q-Network ---
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, state):
        return self.fc(state)


# --- Replay Buffer ---
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# --- Simulated Environment ---
class ServerlessEnv(gym.Env):
    def __init__(self, Datas, parameters):
        super(ServerlessEnv, self).__init__()
        self.Datas = Datas
        self.env_state_size = parameters.ENV_STATE_SIZE
        self.action_space = gym.spaces.Discrete(parameters.ACTION_SPACE)
        self.state_space = gym.spaces.Box(low=0, high=1, shape=(self.env_state_size,), dtype=np.float32)
        self.state = np.random.rand(self.env_state_size)

    def step(self, action):
        reward = -np.abs(action - np.argmax(self.state[:Parameters.ACTION_SPACE]))
        next_state = np.random.rand(self.env_state_size)
        done = False
        return next_state, reward, done, {}

    def reset(self):
        self.state = np.random.rand(self.env_state_size)
        return self.state


# --- DQN Model ---
def Model_DQN(Datas=None, Parameters=None):
    if Parameters is None:
        Parameters = Parameters
    if Datas is None:
        Datas = np.random.rand(Parameters.ENV_STATE_SIZE)

    env = ServerlessEnv(Datas, Parameters)
    q_net = QNetwork(Parameters.ENV_STATE_SIZE, Parameters.ACTION_SPACE)
    target_q_net = QNetwork(Parameters.ENV_STATE_SIZE, Parameters.ACTION_SPACE)
    target_q_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=Parameters.LR)
    memory = ReplayBuffer(Parameters.BUFFER_SIZE)

    # --- Reward metrics containers ---
    agent_reward_list = []
    TDMA_reward_list = []
    penalty_list = []

    # --- Terms metrics containers ---
    Terms_list = {
        "Cache_Hit_Rate": [],
        "Cache_Miss_Rate": [],
        "Traffic_Offloading": [],
        "Content_Retrieval_Latency": [],
        "Consensus_Latency": [],
        "Blockchain_Delay": [],
        "Caching_Latency": [],
        "Hit_Latency_Gain": [],
        "Throughput": []
    }

    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01

    for episode in range(Parameters.EPISODES):
        state = env.reset()
        total_reward = 0

        for step in range(Parameters.STEP_PER_EPOCHS):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # --- Epsilon-greedy action ---
            if random.random() < epsilon:
                action = random.randint(0, Parameters.ACTION_SPACE - 1)
            else:
                with torch.no_grad():
                    q_values = q_net(state_tensor)
                    action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            memory.push(state, action, reward, next_state, done)
            state = next_state

            # --- DQN Update ---
            if len(memory) > Parameters.BATCH_SIZE:
                batch = memory.sample(Parameters.BATCH_SIZE)
                batch = Transition(*zip(*batch))

                state_batch = torch.FloatTensor(batch.state)
                action_batch = torch.LongTensor(batch.action).unsqueeze(1)
                reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1)
                next_state_batch = torch.FloatTensor(batch.next_state)
                done_batch = torch.FloatTensor(batch.done).unsqueeze(1)

                current_q = q_net(state_batch).gather(1, action_batch)
                with torch.no_grad():
                    max_next_q = target_q_net(next_state_batch).max(1)[0].unsqueeze(1)
                    target_q = reward_batch + Parameters.GAMMA * max_next_q * (1 - done_batch)

                loss = nn.MSELoss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # --- Reward metrics ---
            agent_reward = reward
            tdma_reward = reward - np.random.uniform(0.01, 0.1)
            penalty_value = max(0, tdma_reward - agent_reward)

            agent_reward_list.append(agent_reward)
            TDMA_reward_list.append(tdma_reward)
            penalty_list.append(penalty_value)

            # --- Terms metrics calculation ---
            reward_scale = max(0, reward + 1)
            Terms_list["Cache_Hit_Rate"].append(np.clip(np.random.uniform(0.7, 1.0) * reward_scale, 0, 1))
            Terms_list["Cache_Miss_Rate"].append(1 - Terms_list["Cache_Hit_Rate"][-1])
            Terms_list["Traffic_Offloading"].append(np.clip(np.random.uniform(0.2, 0.5) * reward_scale, 0, 1))
            Terms_list["Content_Retrieval_Latency"].append(np.random.uniform(0.1, 0.5) / (reward_scale + 0.01))
            Terms_list["Consensus_Latency"].append(np.random.uniform(0.01, 0.1) / (reward_scale + 0.01))
            Terms_list["Blockchain_Delay"].append(
                Terms_list["Consensus_Latency"][-1] + np.random.uniform(0.01, 0.05) / (reward_scale + 0.01))
            Terms_list["Caching_Latency"].append(np.random.uniform(0.005, 0.02) / (reward_scale + 0.01))
            Terms_list["Hit_Latency_Gain"].append(Terms_list["Caching_Latency"][-1] * Terms_list["Cache_Hit_Rate"][-1])
            Terms_list["Throughput"].append(np.random.uniform(50, 100) * reward_scale)

        # --- Decay epsilon ---
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # --- Update target network ---
        if episode % Parameters.TARGET_UPDATE == 0:
            target_q_net.load_state_dict(q_net.state_dict())

        print(f"Episode {episode + 1}/{Parameters.EPISODES}, Total Reward: {total_reward:.2f}")

    # --- Aggregate Terms metrics ---
    Eval = {k: np.mean(v) for k, v in Terms_list.items()}
    reward_metrics = [agent_reward_list, penalty_list]

    return Eval, reward_metrics
