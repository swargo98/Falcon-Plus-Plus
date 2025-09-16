import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gym
from gym import spaces
import matplotlib.pyplot as plt
import random
from queue import PriorityQueue
from config_sender import configurations
from fpp_simulator import NetworkSystemSimulator, SimulatorState

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def save_model(agent, filename_policy, filename_value):
    torch.save(agent.policy.state_dict(), filename_policy)
    torch.save(agent.value_function.state_dict(), filename_value)
    print("Model saved successfully.")


def load_model(agent, filename_policy, filename_value):
    agent.policy.load_state_dict(torch.load(filename_policy))
    agent.policy_old.load_state_dict(agent.policy.state_dict())
    agent.value_function.load_state_dict(torch.load(filename_value))
    print("Model loaded successfully.")

import math

class NetworkOptimizationEnv(gym.Env):
    def __init__(self, simulator=None):
        super(NetworkOptimizationEnv, self).__init__()
        oneGB = 1024
        self.simulator = NetworkSystemSimulator(network_throughput_per_thread=75,
                                                network_bandwidth=1*oneGB,
                                                track_states=True)
        if simulator is not None:
            self.simulator = simulator
        self.thread_limits = [1, 30]  # Threads can be between 1 and 10

        # Continuous action space: adjustments between -5.0 and +5.0
        self.action_space = spaces.Box(low=np.array([self.thread_limits[0]]),
                               high=np.array([self.thread_limits[1]]),
                               dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array([0, self.thread_limits[0]]),
            high=np.array([
                np.inf,
                self.thread_limits[1]
            ]),
            dtype=np.float32
        )

        self.state = SimulatorState(network_thread=1, network_throughput=0)
        self.max_steps = 10
        self.current_step = 0

        # For recording the trajectory
        self.trajectory = []

    def step(self, action):
        new_thread_counts = np.clip(np.round(action), self.thread_limits[0], self.thread_limits[1]).astype(np.int32)

        # Compute utility and update state
        utility, self.state = self.simulator.get_utility_value(new_thread_counts)

        # Penalize actions that hit thread limits
        penalty = 0
        if new_thread_counts[0] == self.thread_limits[0] or new_thread_counts[0] == self.thread_limits[1]:
            penalty -= 100  # Adjust penalty value as needed

        # Adjust reward
        reward = utility + penalty

        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Record the state
        self.trajectory.append(self.state.copy())

        # Return state as NumPy array
        return self.state.to_array(), reward, done, {}
    
    def reset(self, simulator=None):
        if simulator is not None:
            self.simulator = simulator
            
        self.simulator.network_thread = np.random.randint(3, self.thread_limits[1]-1)

        self.state = SimulatorState(
            network_thread=self.simulator.network_thread,
        )
        
        self.current_step = 0
        self.trajectory = [self.state.copy()]

        # Return initial state as NumPy array
        return self.state.to_array()

class ResidualBlock(nn.Module):
    def __init__(self, size, activation=nn.ReLU):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)
        self.activation = activation()

    def forward(self, x):
        # Save the input (for the skip connection)
        residual = x
        
        # Pass through two linear layers with activation
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        
        # Add the original input (residual connection)
        out += residual
        
        # Optionally add another activation at the end
        out = self.activation(out)
        return out
    
class PolicyNetworkContinuous(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetworkContinuous, self).__init__()
        self.input_layer = nn.Linear(state_dim, 256)
        
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256)
            ) for _ in range(action_dim)
        ])
        
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.to(device)
        
    def forward(self, state):
        x = torch.tanh(self.input_layer(state))
        
        # Residual connections
        for block in self.residual_blocks:
            residual = x
            x = block(x)
            x = torch.tanh(x + residual)
        
        mean = self.mean_layer(x)
        log_std = torch.clamp(self.log_std, -20, 2)
        std = torch.exp(log_std)
        return mean, std

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc_in = nn.Linear(state_dim, 256)
        
        # Add a few residual blocks
        self.res_block1 = ResidualBlock(256, activation=nn.Tanh)
        self.res_block2 = ResidualBlock(256, activation=nn.Tanh)

        # Output value layer
        self.fc_out = nn.Linear(256, 1)
        self.to(device)

    def forward(self, state):
        x = self.fc_in(state)
        x = torch.tanh(x)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        value = self.fc_out(x)
        return value

class PPOAgentContinuous:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, eps_clip=0.2):
        self.policy = PolicyNetworkContinuous(state_dim, action_dim)
        self.policy_old = PolicyNetworkContinuous(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_function = ValueNetwork(state_dim)
        self.optimizer = optim.Adam([
            {'params': self.policy.parameters(), 'lr': lr},
            {'params': self.value_function.parameters(), 'lr': lr}
        ])
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        mean, std = self.policy_old(state)
        dist = Normal(mean, std)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach().cpu().numpy(), action_logprob.detach().cpu().numpy()

    def update(self, memory):
        states = torch.stack(memory.states).to(device)
        actions = torch.tensor(np.array(memory.actions), dtype=torch.float32).to(device)
        rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(device)
        old_logprobs = torch.tensor(np.array(memory.logprobs), dtype=torch.float32).to(device)

        # Compute discounted rewards
        returns = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Get new action probabilities
        mean, std = self.policy(states)
        dist = Normal(mean, std)
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()

        logprobs = logprobs.sum(dim=1)
        old_logprobs = old_logprobs.sum(dim=1)
        entropy = entropy.sum(dim=1)

        ratios = torch.exp(logprobs - old_logprobs)
        state_values = self.value_function(states).squeeze()

        # Compute advantage
        advantages = returns - state_values.detach()

        # Surrogate loss
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - 0.1 * entropy

        # Update policy
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]


from tqdm import tqdm

def train_ppo(env, agent, max_episodes=1000, optimal_reward=0):
    memory = Memory()
    total_rewards = []
    best_avg_reward = 0
    reward_flag = False
    idle_episode = 0
    for episode in tqdm(range(1, max_episodes + 1), desc="Episodes"):
        state = env.reset()
        episode_reward = 0
        for t in range(env.max_steps):
            action, action_logprob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            memory.states.append(torch.FloatTensor(state).to(device))
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
            memory.rewards.append(reward)

            state = next_state
            episode_reward += reward

        agent.update(memory)

        # print(f"Episode {episode}\tLast State: {state}\tReward: {reward}")
        with open('episode_rewards_residual_cl_v1_2.csv', 'a') as f:
                f.write(f"Episode {episode}, Last State: {np.round(state[-3:])}, Reward: {reward}\n")

        memory.clear()
        total_rewards.append(episode_reward)
        if episode % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:])/env.max_steps
            print(f"Episode {episode}\tAverage Reward: {avg_reward:.2f}")
            save_model(agent, "models/"+ configurations['model_version'] +"_offline_policy_"+ str(episode) +".pth", "models/"+ configurations['model_version'] +"_offline_value_"+ str(episode) +".pth")
            print("Model saved successfully.")
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                idle_episode = 0
                with open('best_rewards.csv', 'a') as f:
                    f.write(f"Episode {episode}, Reward: {avg_reward}, Optimal Reward: {optimal_reward}\n")
                save_model(agent, "best_models/"+ configurations['model_version'] +"_offline_policy.pth", "best_models/"+ configurations['model_version'] +"_offline_value.pth")
            else:
                idle_episode +=100

            if not reward_flag and avg_reward > 0.9 * optimal_reward:
                reward_flag = True
                with open('best_rewards.csv', 'a') as f:
                    f.write(f"Episode {episode}, Reward: {avg_reward}******FLAG REACHED**********\n")
            if reward_flag and idle_episode>1000:
                with open('best_rewards.csv', 'a') as f:
                    f.write(f"Episode {episode}, Reward: {avg_reward}******BYEEEEEEEEEEE**********\n")
                    f.write(f"Rewards List:\n {total_rewards}\n")

                break


    return total_rewards

def plot_rewards(rewards, title, pdf_file):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.xlim(0, len(rewards))
    # plt.ylim(-1, 1)
    plt.title(title)
    plt.grid(True)
    
    plt.savefig(pdf_file)  
    plt.close()

import os
if __name__ == '__main__':
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('best_models'):
        os.makedirs('best_models')
    
    oneGB = 1024

    # ── NEW: derive parameters straight from the logs ──
    # from log_stats import extract_log_metrics          # if you saved the helper elsewhere
    # metrics = extract_log_metrics(configurations['model_version'])
    # print(f"Metrics extracted: {metrics}")

    # sender_buffer_capacity      = metrics['sender_buffer_capacity']
    # receiver_buffer_capacity    = metrics['receiver_buffer_capacity']

    # read_throughput_per_thread      = metrics['read_throughput_per_thread']
    # network_throughput_per_thread   = metrics['network_throughput_per_thread']
    # write_throughput_per_thread     = metrics['write_throughput_per_thread']

    # read_bandwidth      = metrics['read_bandwidth']
    # network_bandwidth   = metrics['network_bandwidth']
    # write_bandwidth     = metrics['write_bandwidth']


    # bottleneck = min(read_bandwidth, network_bandwidth, write_bandwidth)
    # optimal_read_thread = bottleneck/read_throughput_per_thread
    # optimal_network_thread = bottleneck/network_throughput_per_thread
    # optimal_write_thread = bottleneck/write_throughput_per_thread

    # optimal_reward = bottleneck * (1/(configurations['K']**optimal_read_thread)
    #                                + 1/(configurations['K']**optimal_network_thread)
    #                                + 1/(configurations['K']**optimal_write_thread))

    # simulator = NetworkSystemSimulator(sender_buffer_capacity=sender_buffer_capacity,
    #                                         receiver_buffer_capacity=receiver_buffer_capacity,
    #                                         read_throughput_per_thread=read_throughput_per_thread,
    #                                         network_throughput_per_thread=network_throughput_per_thread,
    #                                         write_throughput_per_thread=write_throughput_per_thread,
    #                                         read_bandwidth=read_bandwidth,
    #                                         network_bandwidth=network_bandwidth,
    #                                         write_bandwidth=write_bandwidth,
    #                                         track_states=True)
    simulator = NetworkSystemSimulator(network_throughput_per_thread=75,
                                            network_bandwidth=1*oneGB,
                                            track_states=True)
    optimal_reward = 1*oneGB * (1/(configurations['K']**(1*oneGB/75)))
    env = NetworkOptimizationEnv(simulator=simulator)
    agent = PPOAgentContinuous(state_dim=2, action_dim=1, lr=1e-4, eps_clip=0.1)
    rewards = train_ppo(env, agent, max_episodes=30000, optimal_reward=optimal_reward)
    
    plot_rewards(rewards, 'PPO Training Rewards', 'training_rewards_'+ configurations['model_version'] +'.pdf')