import torch
import torch.nn as nn
from core.networks import SharedBackbone, GameHead

class MultiGameNetwork(nn.Module):
    """Network with shared backbone and game-specific heads"""
    
    def __init__(self, input_shape, game_action_spaces, feature_dim = 512):
        super(MultiGameNetwork, self).__init__()

        # Create shared backbone
        self.shared_backbone = SharedBackbone(input_shape, feature_dim)

        # Create game-specific heads
        self.game_heads = nn.ModuleDict()
        for game, n_actions in game_action_spaces.items():
            self.game_heads[game] = GameHead(feature_dim, n_actions)
        
        self.input_shape = input_shape
        self.feature_dim = feature_dim
        self.game_list = list(game_action_spaces.keys())
    
    def forward(self, x, game):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor.
            game (str): The game for which to compute the output.
        
        Returns:
            torch.Tensor: Output tensor from the game-specific head.
        """
        # Pass through shared backbone
        features = self.shared_backbone(x)
        # Pass through game-specific head
        return self.game_heads[game](features)

    def forward_multi(self, inputs_dict):
        """Forward pass for multiple games at once"""
        results = {}
        
        # Process each game
        for game, inputs in inputs_dict.items():
            # Extract features
            features = self.backbone(inputs)
            # Get game-specific output
            results[game] = self.heads[game](features)
            
        return results