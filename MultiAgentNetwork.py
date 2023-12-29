import torch.nn as nn
import torch.optim as optim

class SimpleNetwork(nn.Module):
    """
    Simple neural network with one hidden layer and ReLU activation.

    Parameters:
    - input_dim (int): Dimensionality of the input state.
    - fc1_dim (int): Number of neurons in the first hidden layer.
    - n_action (int): Number of possible actions.
    - lr (float): Learning rate for the optimizer.
    - loss (nn.Module): Loss function for the network.
    """
    def __init__(self, input_dim, fc1_dim, n_action, lr, loss):
        super(SimpleNetwork, self).__init__()
        self.network = nn.Sequential(
              nn.Linear(input_dim, fc1_dim),
              nn.ReLU(),
              nn.Linear(fc1_dim, n_action),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = loss
        
    def forward(self, state):
        actions = self.network(state)
        return actions


class MoreLayersNetwork(nn.Module):
    """
    Neural network with multiple hidden layers and ReLU activation.

    Parameters:
    - input_dim (int): Dimensionality of the input state.
    - fc1_dim (int): Number of neurons in the first hidden layer.
    - fc2_dim (int): Number of neurons in the second hidden layer.
    - fc3_dim (int): Number of neurons in the third hidden layer.
    - n_action (int): Number of possible actions.
    - lr (float): Learning rate for the optimizer.
    - loss (nn.Module): Loss function for the network.
    """
    def __init__(self, input_dim, fc1_dim, fc2_dim, fc3_dim, n_action, lr, loss):
        super(MoreLayersNetwork, self).__init__()
        self.network = nn.Sequential(
              nn.Linear(input_dim, fc1_dim),
              nn.ReLU(),
              nn.Linear(fc1_dim, fc2_dim),
              nn.ReLU(),
              nn.Linear(fc2_dim, fc3_dim),
              nn.ReLU(),
              nn.Linear(fc3_dim, n_action),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = loss
        
    def forward(self, state):
        actions = self.network(state)
        return actions

class SimpleNetworkWithDifferentOptimizer(nn.Module):
    """
    Simple neural network with one hidden layer, ReLU activation, and a different optimizer (Adagrad).

    Parameters:
    - input_dim (int): Dimensionality of the input state.
    - fc1_dim (int): Number of neurons in the first hidden layer.
    - n_action (int): Number of possible actions.
    - lr (float): Learning rate for the optimizer.
    - loss (nn.Module): Loss function for the network.
    """
    def __init__(self, input_dim, fc1_dim, n_action, lr, loss):
        super(SimpleNetworkWithDifferentOptimizer, self).__init__()
        self.network = nn.Sequential(
              nn.Linear(input_dim, fc1_dim),
              nn.ReLU(),
              nn.Linear(fc1_dim, n_action),
        )
        # optimizer with Adagrad
        self.optimizer = optim.Adagrad(self.parameters(), lr=lr)
        self.loss = loss
        
    def forward(self, state):
        actions = self.network(state)
        return actions

class MoreLayersNetworkDifferentOptimizer(nn.Module):
    """
    Neural network with multiple hidden layers, ReLU activation, and a different optimizer (Adagrad).

    Parameters:
    - input_dim (int): Dimensionality of the input state.
    - fc1_dim (int): Number of neurons in the first hidden layer.
    - fc2_dim (int): Number of neurons in the second hidden layer.
    - fc3_dim (int): Number of neurons in the third hidden layer.
    - n_action (int): Number of possible actions.
    - lr (float): Learning rate for the optimizer.
    - loss (nn.Module): Loss function for the network.
    """
    def __init__(self, input_dim, fc1_dim, fc2_dim, fc3_dim, n_action, lr, loss):
        super(MoreLayersNetworkDifferentOptimizer, self).__init__()
        self.network = nn.Sequential(
              nn.Linear(input_dim, fc1_dim),
              nn.ReLU(),
              nn.Linear(fc1_dim, fc2_dim),
              nn.ReLU(),
              nn.Linear(fc2_dim, fc3_dim),
              nn.ReLU(),
              nn.Linear(fc3_dim, n_action),
        )
        # optimizer with Adagrad
        self.optimizer = optim.Adagrad(self.parameters(), lr=lr)
        self.loss = loss

    def forward(self, state):
        actions = self.network(state)
        return actions

class SimpleDifferentLossFunction(nn.Module):
    """
    Simple neural network with one hidden layer, ReLU activation, and a different loss function.

    Parameters:
    - input_dim (int): Dimensionality of the input state.
    - fc1_dim (int): Number of neurons in the first hidden layer.
    - n_action (int): Number of possible actions.
    - lr (float): Learning rate for the optimizer.
    - loss (nn.Module): Different loss function for the network.
    """
    def __init__(self, input_dim, fc1_dim, n_action, lr, loss):
        super(SimpleDifferentLossFunction, self).__init__()
        self.network = nn.Sequential(
              nn.Linear(input_dim, fc1_dim),
              nn

.ReLU(),
              nn.Linear(fc1_dim, n_action),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = loss
        
    def forward(self, state):
        actions = self.network(state)
        return actions

class MoreLayerDifferentLossFunction(nn.Module):
    """
    Neural network with multiple hidden layers, ReLU activation, and a different loss function.

    Parameters:
    - input_dim (int): Dimensionality of the input state.
    - fc1_dim (int): Number of neurons in the first hidden layer.
    - fc2_dim (int): Number of neurons in the second hidden layer.
    - fc3_dim (int): Number of neurons in the third hidden layer.
    - n_action (int): Number of possible actions.
    - lr (float): Learning rate for the optimizer.
    - loss (nn.Module): Different loss function for the network.
    """
    def __init__(self, input_dim, fc1_dim, fc2_dim, fc3_dim, n_action, lr, loss):
        super(MoreLayerDifferentLossFunction, self).__init__()
        self.network = nn.Sequential(
              nn.Linear(input_dim, fc1_dim),
              nn.ReLU(),
              nn.Linear(fc1_dim, fc2_dim),
              nn.ReLU(),
              nn.Linear(fc2_dim, fc3_dim),
              nn.ReLU(),
              nn.Linear(fc3_dim, n_action),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = loss
        
    def forward(self, state):
        actions = self.network(state)
        return actions

class QNetwork(nn.Module):
    """
    Q-network for reinforcement learning.

    Parameters:
    - input_dim (int): Dimensionality of the input state.
    - fc1_dim (int): Number of neurons in the first hidden layer.
    - n_action (int): Number of possible actions.
    - lr (float): Learning rate for the optimizer.
    - loss (nn.Module): Loss function for the network.
    """
    def __init__(self, input_dim, fc1_dim, n_action, lr, loss):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, fc1_dim),
            nn.ReLU(),
            nn.Linear(fc1_dim, n_action),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = loss
        
    def forward(self, state):
        actions = self.network(state)
        return actions
