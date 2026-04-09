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

    def reset(self):
        self.positions = np.random.uniform(0, 2, (self.n_agents, 2))
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for i in range(self.n_agents):
            own = self.positions[i]
            others = np.delete(self.positions, i, axis=0).flatten()
            obs.append(np.concatenate([own, others, self.target]) / self.world_size)
        return np.array(obs)

    def step(self, actions):
        actions = np.clip(actions, -0.5, 0.5)
        self.positions += actions
        self.positions = np.clip(self.positions, 0, self.world_size)

        rewards = []
        done = False

        for i in range(self.n_agents):
            dist = np.linalg.norm(self.positions[i] - self.target)
            reward = -dist

            # Collision penalty
            for j in range(self.n_agents):
                if i != j:
                    d = np.linalg.norm(self.positions[i] - self.positions[j])
                    if d < 1.0:
                        reward -= (1.0 - d) * 5

            rewards.append(reward)

        if np.mean([np.linalg.norm(p - self.target) for p in self.positions]) < 0.5:
            done = True
            rewards = [r + 50 for r in rewards]

        return self._get_obs(), np.array(rewards), done


# ======================
# 🧠 Networks
# ======================
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
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

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = 0.95
        self.tau = 0.01

    def soft_update(self):
        for t, s in zip(self.target_actor.parameters(), self.actor.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

        for t, s in zip(self.target_critic.parameters(), self.critic.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)


# ======================
# 🚀 Training
# ======================
env = DroneSwarmEnv(n_agents=3)

n_agents = 3
obs_dim = 2 + 2*(n_agents-1) + 2
act_dim = 2

agents = [
    MADDPGAgent(obs_dim, act_dim, obs_dim*n_agents, act_dim*n_agents)
    for _ in range(n_agents)
]

buffer = ReplayBuffer()
batch_size = 64

for episode in range(300):
    obs = env.reset()
    total_reward = 0

    for step in range(100):

        # 🎯 Action with noise
        obs_tensor = torch.FloatTensor(obs)
        actions = []
        for i in range(n_agents):
            action = agents[i].actor(obs_tensor[i]).detach().numpy()
            noise = np.random.normal(0, 0.1, size=2)
            actions.append(action + noise)

        next_obs, rewards, done = env.step(actions)

        buffer.push(obs, actions, rewards, next_obs, done)
        obs = next_obs
        total_reward += np.mean(rewards)

        # 🔁 Training
        if len(buffer) > 1000:
            batch_obs, batch_acts, batch_rewards, batch_next_obs, batch_done = buffer.sample(batch_size)

            for i, agent in enumerate(agents):

                obs_all = torch.FloatTensor(batch_obs.reshape(batch_size, -1))
                acts_all = torch.FloatTensor(batch_acts.reshape(batch_size, -1))
                next_obs_all = torch.FloatTensor(batch_next_obs.reshape(batch_size, -1))

                # Target actions
                next_actions = []
                for j in range(n_agents):
                    next_actions.append(
                        agents[j].target_actor(
                            torch.FloatTensor(batch_next_obs[:, j, :])
                        )
                    )
                next_acts_all = torch.cat(next_actions, dim=1)

                # Target Q
                reward = torch.FloatTensor(batch_rewards[:, i]).unsqueeze(1)
                done_mask = torch.FloatTensor(batch_done).unsqueeze(1)

                target_q = reward + agent.gamma * (1 - done_mask) * \
                           agent.target_critic(next_obs_all, next_acts_all)

                # Critic update
                current_q = agent.critic(obs_all, acts_all)
                critic_loss = nn.MSELoss()(current_q, target_q.detach())

                agent.critic_opt.zero_grad()
                critic_loss.backward()
                agent.critic_opt.step()

                # Actor update
                curr_actions = []
                for j in range(n_agents):
                    if j == i:
                        curr_actions.append(
                            agent.actor(torch.FloatTensor(batch_obs[:, j, :]))
                        )
                    else:
                        curr_actions.append(
                            torch.FloatTensor(batch_acts[:, j, :])
                        )

                curr_acts_all = torch.cat(curr_actions, dim=1)
                actor_loss = -agent.critic(obs_all, curr_acts_all).mean()

                agent.actor_opt.zero_grad()
                actor_loss.backward()
                agent.actor_opt.step()

                agent.soft_update()

        if done:
            break

    print(f"Episode {episode}, Reward: {total_reward:.2f}")