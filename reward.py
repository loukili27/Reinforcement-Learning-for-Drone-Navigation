import numpy as np

def compute_reward(num_agents, old_positions, agent_positions, evacuated_agents, deactivated_agents, goal_area):
    rewards = np.zeros(num_agents)

    # Compute reward for each agent
    for i, (old_pos, new_pos) in enumerate(zip(old_positions, agent_positions)):
        if i in evacuated_agents:
            # Already evacuated agents don't get additional rewards
            continue
        elif i in deactivated_agents:
            #  penalty for deactivation
            rewards[i] = -10
        elif tuple(new_pos) in goal_area:
            # Large reward for reaching the goal
            rewards[i] = 1000
            evacuated_agents.add(i)  # Mark the agent as evacuated
        else:
            # Default small penalty per step to encourage efficiency
            rewards[i] = -0.1
            
            # Calculate distance to goal (Euclidean distance)
            goal_x, goal_y = next(iter(goal_area))  # Assume the first goal position is representative
            old_distance = np.linalg.norm(np.array(old_pos) - np.array([goal_x, goal_y]))
            new_distance = np.linalg.norm(np.array(new_pos) - np.array([goal_x, goal_y]))
            
            # Reward for moving closer to the goal
            distance_reward = old_distance - new_distance
            rewards[i] += 1.0 * distance_reward
            
            # Small bonus for exploration (moving to a new position)
            if old_pos[0] != new_pos[0] or old_pos[1] != new_pos[1]:
                rewards[i] += 0.05

    return rewards, evacuated_agents