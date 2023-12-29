import torch 
import torch.nn as nn
import torch.optim as optim
from  ReplayBufferMultiAgent import ReplayBuffer
from MultiAgentNetwork import Qnetwork, SimpleNetwork, SimpleDiffrentLossFunction, SimpleNetworkWithDiffrentOptimizer, MoreLayerDiffrentLossFunction, MoreLayersNetwork, MoreLayersNetworkDiffrentOptimizer
import random 
import os 

class Agent:
    """
    Multi-agent reinforcement learning agent.

    Parameters:
    - input_dimlist (list): List of input dimensions for each agent.
    - fc1_dimlist (list): List of the number of neurons in the first hidden layer for each agent.
    - fc2_dimlist (list): List of the number of neurons in the second hidden layer for each agent.
    - fc3_dimlist (list): List of the number of neurons in the third hidden layer for each agent.
    - fc4_dimlist (list): List of the number of neurons in the fourth hidden layer for each agent.
    - n_actions (int): Number of possible actions for each agent.
    - lrlist (list): List of learning rates for each agent.
    - losslist (list): List of loss functions for each agent.
    - batch_size (list): List of batch sizes for each agent.
    - mem_size (list): List of memory sizes for each agent.
    - gamma_list (list): List of discount factors (gamma) for each agent.
    - num_agents (int): Number of agents.
    - saved_path_list (list): List of paths to pre-trained models for each agent.
    """
    def __init__(self, input_dimlist, fc1_dimlist, fc2_dimlist, fc3_dimlist, fc4_dimlist, n_actions, lrlist, losslist,  batch_size, mem_size, gamma_list, num_agents, saved_path_list):
        self.num_agents = num_agents
        self.evaluate  = False 
        self.agents = []
        self.input_dimlist = input_dimlist
        self.fc1_dimlist = fc1_dimlist
        self.fc2_dimlist = fc2_dimlist
        self.fc3_dimlist = fc3_dimlist
        self.fc4_dimlist = fc4_dimlist
        self.n_actions  = n_actions
        self.lrlist = lrlist
        self.losslist = losslist
        Networks_list = [Qnetwork, MoreLayerDiffrentLossFunction, SimpleNetworkWithDiffrentOptimizer, MoreLayerDiffrentLossFunction, MoreLayersNetwork, MoreLayersNetworkDiffrentOptimizer]
        self.gamma_list = gamma_list

        for index in range(num_agents):
            input_dim = input_dimlist[index]
            fc1_dim = fc1_dimlist[index]
            fc2_dim = fc2_dimlist[index]
            fc3_dim = fc3_dimlist[index]
            fc4_dim = fc4_dimlist[index]
            lr = lrlist[index]
            loss = losslist[index]
            memomry_size = mem_size[index]

            agent_mem = ReplayBuffer(memomry_size, input_dim, n_actions)

            #every agent will  have his own network  
            if saved_path_list[index] == "": 
                agent_network = Qnetwork(input_dim, fc1_dim, fc2_dim, fc3_dim, fc4_dim, n_actions, lr, loss)
            else: 
                # Load the model from the provided saved_path_list for each agent
                agent_network = self.load_model(saved_path_list[index], index)

            gamma = gamma_list[index]

            agent = {
                'mem': agent_mem,
                'network': agent_network,
                'epsilon': 0,
                'n_games': 0,
                'gamma': gamma  # Assign gamma value to agent
            }

            self.agents.append(agent)
            self.batch_size = batch_size

    def choose_action(self, states, validate=False):
        """
        Choose an action for each agent based on the current state.

        Parameters:
        - states (list): List of current states for each agent.
        - validate (bool): Flag indicating whether to use epsilon-greedy exploration during validation.

        Returns:
        - actions (list): List of chosen actions for each agent.
        """
        actions = []

        for agent_index, agent in enumerate(self.agents):
            epsilon = 1000 - agent['n_games']
            final_move = [0, 0, 0]

            if validate == False: 
                if random.randint(0, 1000) < epsilon:
                    move = random.randint(0, 2)
                    final_move[move] = 1
                else:
                    state = states[agent_index]
                    state_tensor = torch.tensor(state, dtype=torch.float)
                    prediction = agent['network'](state_tensor)            
                    move = torch.argmax(prediction).item()
                    final_move[move] = 1
            else: 
                state = states[agent_index]
                state_tensor = torch.tensor(state, dtype=torch.float)
                prediction = agent['network'](state_tensor)            
                move = torch.argmax(prediction).item()
                final_move[move] = 1

            actions.append(final_move)
            
        return actions

    def short_mem(self, states, next_states, actions, rewards, dones):
        """
        Store the current transition in the short-term memory of each agent.

        Parameters:
        - states (list): List of current states for each agent.
        - next_states (list): List of next states for each agent.
        - actions (list): List of chosen actions for each agent.
        - rewards (list): List of rewards received by each agent.
        - dones (list): List of flags indicating whether each agent has finished an episode.
        """
        for agent_index, agent in enumerate(self.agents):
            agent['mem'].store_transition(states[agent_index], next_states[agent_index],
                                          actions[agent_index], rewards[agent_index], dones[agent_index])
            agent['n_games'] += 1
            agent['epsilon'] = 100 - agent['n_games']

        self.learn()

    def long_mem(self):
        """Update the Q-networks of each agent using experiences from the long-term memory."""
        for index, agent in enumerate(self.agents):
            if self.batch_size[index] < agent['mem'].mem_cntr:
                self.learn()

    def long_memory(self, AgentIndex):
        """
        Update the Q-network of a specific agent using experiences from its long-term memory.

        Parameters:
        - AgentIndex (int): Index of the agent to update.
        """
        agent = self.agents[AgentIndex]
        if self.batch_size[AgentIndex] < agent['mem'].mem_cntr:
            self.learn()

    def save(self, agent_idx, Zeitpunkt):
        """
        Save the Q-network of a specific agent.

        Parameters:
        - agent_idx (int): Index of the agent to save.
        - Zeitpunkt (str): Timestamp for identifying the save file.
        """
        file_name = f'Agent{agent_idx}TakenAt{Zeitpunkt}.pth'
        model_folder_path = f'./SavedModells/{agent_idx}'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        agent = self.agents[agent_idx]
        file_name_agent = f'{file_name}_agent_{agent_idx}'
        torch.save(agent['network'].state_dict(), os.path.join(model_folder_path, file_name_agent))

    def load_model(self, saved_path, agent_idx):
        """
        Load a pre-trained model for a specific agent.

        Parameters:
        - saved_path (str): Path to the saved model file.
        - agent_idx (int): Index of the agent.

        Returns:
        - model (Qnetwork): Loaded Q-network model.
        """
        model_folder_path = f'./SavedModells/{agent_idx}'
        file_path = os.path.join(model_folder_path, saved_path)

        # Create an instance of the model that matches the architecture of Qnetwork
        model = Qnetwork(input_dim=self.input_dimlist[agent_idx], 
                        fc1_dim=self.fc1_dimlist[agent_idx],
                        fc2_dim=self.fc2_dimlist[agent_idx],
                        fc3_dim=self.fc3_dimlist[agent_idx],
                        fc4_dim=self.fc4_dimlist[agent_idx],
                        n_action=self.n_actions,
                        lr=self.lrlist[agent_idx],
                        loss=self.losslist[agent_idx])

        # Load the pre-trained weights into the model
        model.load_state_dict(torch.load(file_path))
        return model

    def learn(self):
        """Update the Q-networks of each agent based on experiences sampled from their long-term memory."""
        for agent_index, agent in enumerate(self.agents):
            states, next_states, actions, rewards, dones = agent['mem'].sample_batch(self.batch_size[agent_index])

            state_tensor = torch.tensor(states, dtype=torch.float)
            next_state_tensor = torch.tensor(next_states, dtype=torch.float)
            action_tensor = torch.tensor(actions, dtype=torch.long)
            reward_tensor = torch.tensor(rewards, dtype=torch.float)
            done_tensor = torch.tensor(dones)

            if len(state_tensor.shape) == 1:
                state_tensor = torch.unsqueeze(state_tensor, 0)
                next_state_tensor = torch.unsqueeze(next_state_tensor, 0)
                action_tensor = torch.unsqueeze(action_tensor, 0)
                reward_tensor = torch.unsqueeze(reward_tensor, 0)
                done_tensor = torch.unsqueeze(done_tensor, 0)

            pred = agent['network'](state_tensor)

            target = pred.clone()
            for idx in range(len(done_tensor)):
                Q_new = reward_tensor[idx]
                if not done_tensor[idx]:
                    Q_new = reward_tensor[idx] + agent['gamma'] * torch.max(agent['network'](next_state_tensor[idx]))

                target[idx][torch.argmax(action_tensor[idx]).item()] = Q_new

            agent['network'].optimizer.zero_grad()
            loss = agent['network'].loss(target, pred)
            loss.backward()
            agent['network'].optimizer.step()
