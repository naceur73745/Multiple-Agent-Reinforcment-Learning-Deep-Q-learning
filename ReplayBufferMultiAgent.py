import numpy as np 


class ReplayBuffer:

    """
    ReplayBuffer class for storing and sampling transitions for reinforcement learning.

    Parameters:
    - capacity (int): Maximum number of transitions to store.
    - input_dim (int): Dimensionality of the input state.
    - n_actions (int): Number of possible actions.

    Attributes:
    - capacity (int): Maximum capacity of the replay buffer.
    - input_dim (int): Dimensionality of the input state.
    - n_actions (int): Number of possible actions.
    - mem_cntr (int): Counter for the number of stored transitions.
    - states (numpy.ndarray): Array to store states of shape (capacity, input_dim).
    - next_states (numpy.ndarray): Array to store next states of shape (capacity, input_dim).
    - actions (numpy.ndarray): Array to store actions of shape (capacity, n_actions).
    - rewards (numpy.ndarray): Array to store rewards of shape (capacity).
    - dones (numpy.ndarray): Array to store done flags of shape (capacity, bool).
    """

    def __init__(self, capacity, input_dim, n_actions):
        self.capacity = capacity
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.mem_cntr = 0
        self.states = np.zeros((self.capacity, self.input_dim))
        self.next_states = np.zeros((self.capacity, self.input_dim))
        self.actions = np.zeros((self.capacity, self.n_actions))
        self.rewards = np.zeros(self.capacity)
        self.dones = np.zeros(self.capacity, dtype=bool)

    def store_transition(self, state, next_state, action, reward, done):
        """
        Store a transition in the replay buffer.

        Parameters:
        - state (numpy.ndarray): Current state.
        - next_state (numpy.ndarray): Next state.
        - action (numpy.ndarray): Action taken.
        - reward (float): Reward received.
        - done (bool): Whether the episode terminated after this transition.
        """
        index = self.mem_cntr % self.capacity
        self.states[index] = state
        self.next_states[index] = next_state
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done
        self.mem_cntr += 1

    def sample_batch(self, batch_size):
        """
        Sample a batch of transitions from the replay buffer.

        Parameters:
        - batch_size (int): Size of the batch to sample.

        Returns:
        - Tuple of numpy arrays: Sampled batch of states, next_states, actions, rewards, and dones.
        """
        max_mem = min(self.mem_cntr, self.capacity)

        if max_mem >= batch_size:
             batch_indices = np.random.choice(max_mem, batch_size, replace=False)
        else:
             batch_indices = np.random.choice(max_mem, batch_size, replace=True)

        states = self.states[batch_indices]
        next_states = self.next_states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        dones = self.dones[batch_indices]

        return states, next_states, actions, rewards, dones
