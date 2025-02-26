import numpy as np

class MyAgent():
    def __init__(self, num_agents: int, alpha=0.1, gamma=0.9, epsilon=0.2, lambda_=0.8):
        self.rng = np.random.default_rng()
        self.num_agents = num_agents
        
        # Hyperparamètres du SARSA(λ)
        self.alpha = alpha  # Taux d'apprentissage
        self.gamma = gamma  # Facteur de discount
        self.epsilon = epsilon  # Probabilité d'exploration
        self.lambda_ = lambda_  # Facteur d’oubli des traces
        
        # Q-table et traces d’éligibilité
        self.q_table = {}
        self.e_table = {}  # Traces d'éligibilité

    def get_action(self, state: list, evaluation: bool = False):
        actions = []
        for agent_idx in range(self.num_agents):
            state_tuple = tuple(state[agent_idx])  # Convertir en clé (immuable)

            if state_tuple not in self.q_table:
                self.q_table[state_tuple] = np.zeros(7)  # Initialisation

            if evaluation or self.rng.random() > self.epsilon:
                action = np.argmax(self.q_table[state_tuple])  # Exploitation
            else:
                action = self.rng.integers(0, 7)  # Exploration

            actions.append(action)
        return actions

    def update_policy(self, state, actions, rewards, next_state, next_actions):
        for agent_idx in range(self.num_agents):
            state_tuple = tuple(state[agent_idx])
            next_state_tuple = tuple(next_state[agent_idx])

            # Vérifier et initialiser Q(s, a) si nécessaire
            if state_tuple not in self.q_table:
                self.q_table[state_tuple] = np.zeros(7)
            if next_state_tuple not in self.q_table:
                self.q_table[next_state_tuple] = np.zeros(7)  # Nouvelle initialisation

            # Extraction de la récompense du bon agent
            reward = rewards[agent_idx]  # On récupère la récompense spécifique

            # SARSA(λ) mise à jour
            self.q_table[state_tuple][actions[agent_idx]] += self.alpha * (
                reward + self.gamma * self.q_table[next_state_tuple][next_actions[agent_idx]]
                - self.q_table[state_tuple][actions[agent_idx]]
            )


import torch
import torch.nn as nn
import torch.optim as optim
import random

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)

class MyAgentDQN:
    def __init__(self, state_dim, num_actions, gamma=0.9, epsilon=0.1, lr=0.001):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.model = DQN(state_dim, num_actions)
        self.target_model = DQN(state_dim, num_actions)  # Modèle cible
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = []  # Replay buffer
        self.batch_size = 32

    def get_action(self, state, evaluation=False):
        if not evaluation and random.random() < self.epsilon:
            return [random.randint(0, self.num_actions - 1) for _ in range(len(state))]
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                q_values = self.model(state_tensor)
                return [torch.argmax(q).item() for q in q_values]

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def update_policy(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_tensor = torch.FloatTensor(state_batch)
        action_tensor = torch.LongTensor(action_batch).unsqueeze(1)
        reward_tensor = torch.FloatTensor(reward_batch)
        next_state_tensor = torch.FloatTensor(next_state_batch)
        done_tensor = torch.FloatTensor(done_batch)

        q_values = self.model(state_tensor).gather(1, action_tensor).squeeze()
        with torch.no_grad():
            next_q_values = self.target_model(next_state_tensor).max(1)[0]
            target_q_values = reward_tensor + self.gamma * next_q_values * (1 - done_tensor)

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
