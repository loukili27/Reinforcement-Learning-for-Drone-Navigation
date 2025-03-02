import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import Tuple, Set, List, Dict

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, transition: tuple):
        """Ajoute une transition (état, action, récompense, nouvel état) dans le buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        """Renvoie un échantillon aléatoire de transitions."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class MultiAgentQNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, agent_features: int = 12, other_agent_features: int = 8):
        super(MultiAgentQNetwork, self).__init__()
        self.agent_features = agent_features
        self.other_agent_features = other_agent_features
        
        # Réseau pour traiter les caractéristiques propres de l'agent
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_features, 64),
            nn.ReLU()
        )
        
        # Réseau pour traiter les informations des autres agents
        self.other_agents_encoder = nn.Sequential(
            nn.Linear(other_agent_features, 32),
            nn.ReLU()
        )
        
        # Mécanisme d'attention pour pondérer les autres agents
        self.attention = nn.Sequential(
            nn.Linear(32, 1),
            nn.Softmax(dim=1)
        )
        
        # Réseau de fusion et décision
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # Caractéristiques de l'agent lui-même (les 12 premières features)
        agent_features = x[:, :self.agent_features]
        agent_encoding = self.agent_encoder(agent_features)
        
        # Traitement plus robuste des caractéristiques des autres agents
        if x.size(1) > self.agent_features:
            # Extraire les infos des autres agents
            other_agents_features = x[:, self.agent_features:]
            
            # Calculer combien d'agents on peut extraire complètement
            complete_agents = other_agents_features.size(1) // self.other_agent_features
            
            if complete_agents > 0:
                # Ne prendre que les données complètes d'agents
                usable_features = complete_agents * self.other_agent_features
                reshaped_features = other_agents_features[:, :usable_features].reshape(
                    batch_size, complete_agents, self.other_agent_features)
                
                # Encoder chaque agent
                other_encodings = []
                for i in range(complete_agents):
                    other_encodings.append(self.other_agents_encoder(reshaped_features[:, i, :]))
                
                # Appliquer l'attention si plusieurs agents
                if complete_agents > 1:
                    other_encodings = torch.stack(other_encodings, dim=1)
                    attention_weights = self.attention(other_encodings)
                    other_encoding = torch.sum(attention_weights * other_encodings, dim=1)
                else:
                    other_encoding = other_encodings[0]
            else:
                # Aucun agent complet
                other_encoding = torch.zeros(batch_size, 32, device=x.device)
        else:
            # Aucune info sur d'autres agents
            other_encoding = torch.zeros(batch_size, 32, device=x.device)
        
        # Fusion et décision
        combined = torch.cat([agent_encoding, other_encoding], dim=1)
        return self.fusion(combined)

class MyAgent():
    def __init__(self, 
                 num_agents: int, 
                alpha=1e-3,    # Taux d'apprentissage standard
                gamma=0.9,    # Planification très long terme
                epsilon=1.0,   # Exploration complète au départ
                epsilon_decay=0.997,  # Décroissance modérée
                epsilon_min=0.08,     # Exploration minimale modérée
                buffer_capacity=3000,  # Grande mémoire pour diversité d'expériences
                batch_size=256,       # Grand batch pour apprentissage stable
                target_update_freq=200,  # Mise à jour équilibrée
                communication_range=10.0 ): # Portée maximale pour coordination optimale
        
        self.num_agents = num_agents
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_actions = 7
        self.communication_range = communication_range

        # Définition des dimensions pour le modèle
        self.agent_features = 12  # Position, orientation, status, goal, LIDARs
        self.other_agent_features = 8  # Position relative, orientation, status, LIDAR principal
        
        # Initialisation du buffer de rejouage commun
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Buffer spécifique pour les expériences positives (succès)
        self.success_buffer = ReplayBuffer(1000)

        # État et action précédents pour chaque drone
        self.last_states = [None] * num_agents  
        self.last_actions = [None] * num_agents
        
        # Suivi des performances pour adapter le comportement
        self.agent_success_rate = [0.0] * num_agents
        self.episode_count = 0
        
        # Réseau et optimisation
        self.q_net = None
        self.target_net = None
        self.optimizer = None
        self.target_update_freq = target_update_freq
        self.learn_step = 0

    def initialize_network(self, state: list):
        """Initialise le réseau avec la structure d'état complète"""
        state_dim = len(state[0])
        self.q_net = MultiAgentQNetwork(state_dim, self.n_actions, self.agent_features, self.other_agent_features)
        self.target_net = MultiAgentQNetwork(state_dim, self.n_actions, self.agent_features, self.other_agent_features)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.alpha)

    def get_action(self, state: list, evaluation: bool = False):
        """Sélection d'actions avec prise en compte des autres agents"""
        if self.q_net is None:
            self.initialize_network(state)
        
        actions = []
        for i, s in enumerate(state):
            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            
            # Stratégie exploration/exploitation
            if evaluation or np.random.rand() > self.epsilon:
                q_values = self.q_net(s_tensor)
                action = int(torch.argmax(q_values, dim=1).item())
                
                # Dans un environnement avec plusieurs agents, privilégier les actions qui évitent la congestion
                if not evaluation and self._detect_similar_positions(i, state):
                    # Ajouter une perturbation à l'action pour diversifier les comportements
                    if np.random.rand() < 0.3:  # 30% de chance de diversifier
                        action = (action + np.random.randint(1, self.n_actions)) % self.n_actions
            else:
                action = int(np.random.randint(0, self.n_actions))
            
            actions.append(action)
            self.last_states[i] = s
            self.last_actions[i] = action
            
        return actions
    
    def _detect_similar_positions(self, agent_idx, states):
        """Détecte si des agents sont trop proches les uns des autres"""
        my_pos = states[agent_idx][:2]  # Coordonnées x,y
        for i, s in enumerate(states):
            if i != agent_idx:
                other_pos = s[:2]
                dist = np.sqrt(np.sum((np.array(my_pos) - np.array(other_pos))**2))
                if dist < 2.0:  # Distance de proximité
                    return True
        return False
    
    def update_policy(self, actions: list, state: list, rewards):
        """Mise à jour de la politique avec apprentissage cooperatif"""
        # Enregistrement des transitions
        for i, s_next in enumerate(state):
            if self.last_states[i] is None:
                continue
            
            transition = (self.last_states[i], self.last_actions[i], rewards[i], s_next)
            self.replay_buffer.push(transition)
            
            # Conserver les expériences réussies séparément
            if rewards[i] > 0.5:  # Seuil pour considérer une expérience comme positive
                self.success_buffer.push(transition)
                self.agent_success_rate[i] = 0.9 * self.agent_success_rate[i] + 0.1
            
            self.last_states[i] = s_next

        # Ne faire l'apprentissage que si le buffer contient assez d'échantillons
        if len(self.replay_buffer) < self.batch_size:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            return

        # 1. Apprentissage standard sur un batch aléatoire
        batch = self.replay_buffer.sample(self.batch_size)
        loss = self._learn_from_batch(batch)
        
        # 2. Apprentissage par imitation des expériences réussies (si disponibles)
        if len(self.success_buffer) >= self.batch_size // 2:
            success_batch = self.success_buffer.sample(self.batch_size // 2)
            random_batch = self.replay_buffer.sample(self.batch_size // 2)
            combined_batch = success_batch + random_batch
            self._learn_from_batch(combined_batch)
        
        # Mise à jour d'epsilon et du réseau cible
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.learn_step += 1
        
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
    
    def _learn_from_batch(self, batch):
        """Apprentissage à partir d'un batch de transitions"""
        states, actions, rewards, next_states = zip(*batch)
        
        # Conversion en tenseurs
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1)
        rewards_tensor = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32)
        
        # Double DQN: sélection avec q_net, évaluation avec target_net
        current_q = self.q_net(states_tensor).gather(1, actions_tensor)
        
        with torch.no_grad():
            best_actions = self.q_net(next_states_tensor).max(1)[1].unsqueeze(1)
            next_q = self.target_net(next_states_tensor).gather(1, best_actions)
            target_q = rewards_tensor + self.gamma * next_q
        
        # Calcul de la perte et optimisation
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()