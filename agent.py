import numpy as np

class MyAgent:
    def __init__(self, num_agents: int, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.rng = np.random.default_rng()
        self.num_agents = num_agents
        
        # Hyperparamètres du Q-learning
        self.alpha = alpha  # Taux d'apprentissage
        self.gamma = gamma  # Facteur de discount
        self.epsilon = epsilon  # Probabilité d'exploration
        
        # Table Q : Dictionnaire associant état -> valeurs d'actions
        self.q_table = {}

    def get_action(self, state: list, evaluation: bool = False):
        actions = []
        for agent_idx in range(self.num_agents):
            state_tuple = tuple(state[agent_idx])  # Convertir en clé (immuable)

            if state_tuple not in self.q_table:
                self.q_table[state_tuple] = np.zeros(7)  # Initialisation (ici 7 actions possibles)

            if evaluation or self.rng.random() > self.epsilon:
                # Exploitation : choisir la meilleure action connue
                action = np.argmax(self.q_table[state_tuple])
            else:
                # Exploration : choisir une action aléatoire
                action = self.rng.integers(0, 7)

            actions.append(action)
        return actions

    def update_policy(self, actions: list, state: list, reward: float, next_state: list):
        # Assurez-vous que next_state est une liste de positions pour chaque agent
        for agent_idx in range(self.num_agents):
            state_tuple = tuple(state[agent_idx])
            next_state_tuple = tuple(next_state[agent_idx])

            # Initialisation dans la Q-table si l'état n'existe pas
            if state_tuple not in self.q_table:
                self.q_table[state_tuple] = np.zeros(7)
            if next_state_tuple not in self.q_table:
                self.q_table[next_state_tuple] = np.zeros(7)

            # Q-learning update
            best_next_action = np.argmax(self.q_table[next_state_tuple])
            # Mise à jour selon la règle du Q-learning
            self.q_table[state_tuple][actions[agent_idx]] += self.alpha * (
                reward + self.gamma * self.q_table[next_state_tuple][best_next_action]
                - self.q_table[state_tuple][actions[agent_idx]]
            )
    