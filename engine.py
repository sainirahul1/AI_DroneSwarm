import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# ======================
# 🌍 Environment
# ======================
class DroneSwarmEnv:
    def __init__(self, n_agents=3, world_size=10):
        self.n_agents = n_agents
        self.world_size = world_size
        self.target = np.array([8.0, 8.0])
        self.positions = np.zeros((self.n_agents, 2))
        self.last_dists = np.zeros(self.n_agents)

    def reset(self):
        # Initialize drones far from target for better learning
        self.positions = np.random.uniform(0, 3, (self.n_agents, 2))
        self.last_dists = np.array([np.linalg.norm(p - self.target) for p in self.positions])
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for i in range(self.n_agents):
            own = self.positions[i]
            # Others' relative positions
            others_rel = (np.delete(self.positions, i, axis=0) - own).flatten()
            # Target relative position
            target_rel = (self.target - own).flatten()
            
            # Normalize by world_size
            obs.append(np.concatenate([own/self.world_size, others_rel/self.world_size, target_rel/self.world_size]))
        return np.array(obs)

    def step(self, actions):
        actions = np.clip(actions, -0.8, 0.8) # Slightly faster
        self.positions += actions
        self.positions = np.clip(self.positions, 0, self.world_size)

        rewards = []
        done = False
        reached_count = 0

        for i in range(self.n_agents):
            dist = np.linalg.norm(self.positions[i] - self.target)
            
            # 1. Base distance penalty (minimized)
            reward = -dist * 0.1
            
            # 2. Shaping: Reward for getting closer compared to last step
            diff = self.last_dists[i] - dist
            reward += diff * 5.0 # High sensitivity to progress
            self.last_dists[i] = dist
            
            # 3. Collision penalty
            for j in range(self.n_agents):
                if i != j:
                    d = np.linalg.norm(self.positions[i] - self.positions[j])
                    if d < 0.6: # Smaller collision radius
                        reward -= 5.0

            # 4. Success bonus (reaching target)
            if dist < 1.0: # Reaching threshold
                reward += 20.0
                reached_count += 1

            rewards.append(reward)

        # Global success: All drones within 1.5m or any drone hit target center
        if reached_count == self.n_agents or np.min([np.linalg.norm(p - self.target) for p in self.positions]) < 0.3:
            done = True
            rewards = [r + 100 for r in rewards]

        return self._get_obs(), np.array(rewards), done


# ======================
# 🧠 Networks
# ======================
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, total_obs_dim, total_act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_act_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, acts):
        x = torch.cat([obs, acts], dim=1)
        return self.net(x)


# ======================
# 📦 Replay Buffer
# ======================
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, acts, rewards, next_obs, done):
        self.buffer.append((obs, acts, rewards, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, acts, rewards, next_obs, done = map(np.array, zip(*batch))
        return obs, acts, rewards, next_obs, done

    def __len__(self):
        return len(self.buffer)


# ======================
# 🤖 MADDPG Agent
# ======================
class MADDPGAgent:
    def __init__(self, obs_dim, act_dim, total_obs_dim, total_act_dim):
        self.actor = Actor(obs_dim, act_dim)
        self.critic = Critic(total_obs_dim, total_act_dim)

        self.target_actor = Actor(obs_dim, act_dim)
        self.target_critic = Critic(total_obs_dim, total_act_dim)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=5e-4) # Higher LR
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = 0.95
        self.tau = 0.01

    def soft_update(self):
        for t, s in zip(self.target_actor.parameters(), self.actor.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

        for t, s in zip(self.target_critic.parameters(), self.critic.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)
