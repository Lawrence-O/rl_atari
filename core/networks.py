import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQNNetwork(nn.Module):
    """Deep Q-Network for Atari games"""
    
    def __init__(self, input_shape, num_actions, device="cpu"):
        """Initialize the DQN network
        
        Args:
            input_shape: Shape of input state (channel, height, width)
            num_actions: Number of possible actions
            device: Device to run the network on
        """
        super(DQNNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.device = device
        
        # Convolutional layers - standard architecture from DQN paper
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate the size of the flattened features
        conv_output_size = self._get_conv_output_size(input_shape)
        assert conv_output_size == 7*7*64, "Convolutional output size is incorrect"
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
        # Initialize weights using orthogonal initialization
        self._initialize_weights()
        
    def _get_conv_output_size(self, shape):
        """Calculate the output size of the convolutional layers"""
        o = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def _initialize_weights(self):
        """Initialize network weights with orthogonal initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass through the network"""
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

class DuelingDQNNetwork(nn.Module):
    """Dueling DQN network for Atari games"""
    
    def __init__(self, input_shape, num_actions, device="cpu"):
        """Initialize the dueling DQN network
        
        Args:
            input_shape: Shape of input state (channel, height, width)
            num_actions: Number of possible actions
            device: Device to run the network on
        """
        super(DuelingDQNNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.device = device
        
        # Convolutional layers - standard architecture from DQN paper
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate the size of the flattened features
        conv_output_size = self._get_conv_output_size(input_shape)
        assert conv_output_size == 7*7*64, "Convolutional output size is incorrect"
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
        # Initialize weights using orthogonal initialization
        self._initialize_weights()
        
    def _get_conv_output_size(self, shape):
        """Calculate the output size of the convolutional layers"""
        o = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def _initialize_weights(self):
        """Initialize network weights with orthogonal initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass through the network"""
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Flatten
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class NoisyLinear(nn.Module):
    """Noisy linear layer for NoisyNet"""
    
    def __init__(self, in_features, out_features, std_init=0.5, min_sigma=float(1e-3)):
        """Initialize the noisy linear layer
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            std_init: Initial standard deviation for noise
            min_sigma: Minimum standard deviation for noise
        """
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.min_sigma = min_sigma
        
        # Weight parameters
        # Mean values
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))

        # Standard deviation values
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Reset parameters of the layer"""
        mu_range = 3 / np.sqrt(self.in_features)

        # Initialize weights and biases uniformly
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def _scale_noise(self, size):
        """Generate scaled noise for factorized Gaussian noise"""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def reset_noise(self):
        """Reset noise for the layer"""
        # Generate factorized Gaussian noise
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
    
        # Outer product
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _clamp_sigma(self):
        """Clamp the standard deviation to a minimum value"""
        with torch.no_grad():
            self.weight_sigma.data.clamp_(min=self.min_sigma)
            self.bias_sigma.data.clamp_(min=self.min_sigma)
    
    def forward(self, x):
        """Forward pass through the noisy linear layer"""
        self._clamp_sigma()
        if self.training:
            # During training, use both deterministic and noisy parts
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # During evaluation, use only the deterministic part
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

class DuelingNoisyNetwork(nn.Module):
    """Dueling architecture with NoisyNet layers for exploration"""
    
    def __init__(self, input_shape, num_actions, device="cpu"):
        """Initialize the dueling noisy network
        
        Args:
            input_shape: Shape of input state (channel, height, width)
            num_actions: Number of possible actions
            device: Device to run the network on
        """
        super(DuelingNoisyNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.device = device
        
        # Convolutional layers remain deterministic
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate the size of the flattened features
        conv_output_size = self._get_conv_output_size(input_shape)
        
        # Value stream with noisy layers
        self.value_stream = nn.Sequential(
            NoisyLinear(conv_output_size, 512),
            nn.ReLU(),
            NoisyLinear(512, 1)
        )
        
        # Advantage stream with noisy layers
        self.advantage_stream = nn.Sequential(
            NoisyLinear(conv_output_size, 512),
            nn.ReLU(),
            NoisyLinear(512, num_actions)
        )
        
        # Initialize convolutional weights
        self._initialize_conv_weights()
        
    def _get_conv_output_size(self, shape):
        """Calculate the output size of the convolutional layers"""
        o = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def _initialize_conv_weights(self):
        """Initialize convolutional weights with orthogonal initialization"""
        for m in self.conv_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass through the network"""
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    
    def reset_noise(self):
        """Reset noise for all noisy layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class SharedBackbone(nn.Module):
    """Shared convolutional backbone for muli game representation learning"""

    def __init__(self, input_shape, feature_dim=512):
        """Initialize the shared backbone
        
        Args:
            input_shape: Shape of input state (channel, height, width)
            feature_dim: Dimension of the output feature vector
        """
        super(SharedBackbone, self).__init__()
        
        self.input_shape = input_shape
        self.feature_dim = feature_dim
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_output_size = self._get_conv_output_size(input_shape)
        assert conv_output_size == 7*7*64, "Convolutional output size is incorrect"
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, feature_dim),
            nn.ReLU()
        )

        self._initialize_weights()

    def _get_conv_output_size(self, shape):
        """Calculate the output size of the convolutional layers"""
        o = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        """Forward pass through the shared backbone"""
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    
    def _initialize_weights(self):
        """Initialize network weights with orthogonal initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class GameHead(nn.Module):
    """Game-specific head for the shared backbone"""
    
    def __init__(self, input_dim, output_dim):
        """Initialize the game head
        
        Args:
            input_dim: Dimension of the input feature vector
            output_dim: Dimension of the output feature vector (number of actions)
        """
        super(GameHead, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Use NoisyLinear layers
        self.advntage_stream = nn.Sequential(
            NoisyLinear(input_dim, 512),
            nn.ReLU(),
            NoisyLinear(512, output_dim)
        )
        self.value_stream = nn.Sequential(
            NoisyLinear(input_dim, 512),
            nn.ReLU(),
            NoisyLinear(512, 1)
        )

        self.reset_noise()
     
    def reset_noise(self):
        """Reset noise for all noisy layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def forward(self, x):
        """Forward pass through the game head"""
        advantage = self.advntage_stream(x)
        value = self.value_stream(x)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class PolicyNetwork(nn.Module):
    """PolicyNetwork network for policy gradient methods"""
    
    def __init__(self, input_dim, output_dim, hidden_dim=128, device="cpu"):
        """Initialize the Policy network
        
        Args:
            input_dim: Dimension of the input feature vector
            output_dim: Dimension of the output feature vector
            hidden_dim: Dimension of the hidden layers
            device: Device to run the network on
        """
        super(PolicyNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # Convolutional layers
        if isinstance(input_dim, (tuple, list)) and len(input_dim) == 3:
            self.input_type = 'image'
            self.conv_layers = nn.Sequential(
                nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            conv_output_size = self._get_conv_output_size(input_dim)
            # Fully connected layers
            self.fc_layers = nn.Sequential(
                nn.Linear(conv_output_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.input_type = 'vector'
            # Fully connected layers for vector input
            self.fc_layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
            self.conv_layers = None
        
        # Initialize weights using orthogonal initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights with orthogonal initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):  # Add Conv2d 
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def _get_conv_output_size(self, shape):
        """Calculate output size of convolutional layers"""
        o = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        if self.input_type == 'image':
            x = self.conv_layers(x)
        else:
            x = x.view(-1, self.input_dim)  # Flatten for vector input
        return self.fc_layers(x)

class OptionPolicy(nn.Module):
    """Individual option policy network"""
    
    def __init__(self, state_shape, num_actions, device="cpu",):
        super(OptionPolicy, self).__init__()
        self.device = device
        self.network = DuelingNoisyNetwork(state_shape, num_actions, device)
    
    def forward(self, state):
        return self.network(state)
    def reset_noise(self):
        if hasattr(self.network, 'reset_noise'):
            self.network.reset_noise()
            
        
class TerminationNetwork(nn.Module):
    """Termination network for options"""
    
    def __init__(self, state_shape, num_options, device="cpu"):
        super(TerminationNetwork, self).__init__()
        self.device = device
        self.network = DuelingNoisyNetwork(state_shape, num_options, device)
    
    def forward(self, state):
        return torch.sigmoid(self.network(state)) # returns a continous value between 0 and 1
    
    def reset_noise(self):
        if hasattr(self.network, 'reset_noise'):
            self.network.reset_noise()    

class MetaController(nn.Module):
    """Meta-controller for options"""
    
    def __init__(self, state_shape, num_options, device="cpu"):
        super(MetaController, self).__init__()
        self.device = device
        self.network = DuelingNoisyNetwork(state_shape, num_options, device=device)
    
    def forward(self, state):
       return self.network(state)
        
    
    def reset_noise(self):
        if hasattr(self.network, 'reset_noise'):
            self.network.reset_noise()
