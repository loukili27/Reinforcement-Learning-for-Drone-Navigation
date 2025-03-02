import numpy as np


def compute_reward(num_agents, old_positions, agent_positions, evacuated_agents, deactivated_agents, goal_area):


    rewards = np.zeros(num_agents)
    COLLABORATION_THRESHOLD = 3.0  # Distance pour considérer une collaboration
    OBSTACLE_PENALTY = {
        1: -1,   # Mur/bordure
        2: -2,   # Obstacle dynamique
        3: 0    
    }

    for i, (old_pos, new_pos) in enumerate(zip(old_positions, agent_positions)):
        if i in evacuated_agents:
            continue
        elif i in deactivated_agents:
            rewards[i] = -5
        elif tuple(new_pos) in goal_area:
            rewards[i] = 1000.0
            evacuated_agents.add(i)
        else:
            state = self.get_agent_state(i)
            
            # Données de base
            agent_pos = state[0:2]
            goal_pos = state[4:6]
            lidar_distances = state[[6, 8, 10]]  # Distances dans les 3 directions
            lidar_types = state[[7, 9, 11]]     # Types d'obstacles
            
            # 1. Récompense/pénalité de progression
            old_dist = np.linalg.norm(old_pos - goal_pos)
            new_dist = np.linalg.norm(new_pos - goal_pos)
            rewards[i] += (old_dist - new_dist) * 0.5  # Scale le gain de progression

            # 2. Pénalités pour obstacles proches (avec différentiation par type)
            for dist, obs_type in zip(lidar_distances, lidar_types):
                if obs_type in OBSTACLE_PENALTY and dist < 2.0:
                    rewards[i] += OBSTACLE_PENALTY[obs_type] * (2.0 - dist)

            # 3. Interactions avec les autres agents
            other_agent_index = 12  # Début des données des autres agents
            for _ in range(num_agents - 1):
                if other_agent_index + 10 > len(state):
                    break
                
                # Extraire les données de l'agent communicant
                other_pos = state[other_agent_index:other_agent_index+2]
                other_status = state[other_agent_index+3]
                other_goal = goal_area[_]  # Récupère le goal de l'agent communicant
                
                # a) Éviter les collisions
                distance_to_other = np.linalg.norm(agent_pos - other_pos)
                if distance_to_other < 1.5:
                    rewards[i] -= 0.4
                
                # b) Collaboration vers les objectifs
                if other_status == 0:  # Si l'agent est actif
                    # Récompense si on se rapproche du goal de l'autre agent
                    progress_to_other_goal = np.linalg.norm(other_pos - other_goal) - np.linalg.norm(agent_pos - other_goal)
                    rewards[i] += progress_to_other_goal * 0.2
                
                other_agent_index += 10  # Passer au prochain agent

            # 4. Pénalité d'immobilité
            if np.array_equal(old_pos, new_pos):
                rewards[i] -= 1

    return rewards, evacuated_agents