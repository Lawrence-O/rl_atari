"""
Generic training script for RL agents in Atari environments.
"""

import os
import sys
import argparse
import yaml

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.trainer import RLTrainer
from algorithms.dqn.dqn_agent import DQNAgent
from algorithms.dqn.rainbow_agent import RainbowAgent
from algorithms.policy_gradient._ppo import PPOAgent
from algorithms.policy_gradient.reinforce import ReinforceAgent
from algorithms.dqn.options_rainbow_agent import OptionsRainbowAgent

# Dictionary mapping agent names to agent classes
AGENT_CLASSES = {
    "dqn": DQNAgent,
    "rainbow": RainbowAgent,
    "reinforce": ReinforceAgent,
    "ppo": PPOAgent,
    'hrl': OptionsRainbowAgent,
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a RL agent on Atari games")
    parser.add_argument("--config", type=str, default="experiments/configs/dqn_pong.yaml", 
                      help="Path to the configuration file")
    parser.add_argument("--agent", type=str, default="dqn", choices=AGENT_CLASSES.keys(),
                      help="Agent type to train")
    parser.add_argument("--experiment_name", type=str, default=None,
                      help="Name of the experiment")
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get agent class
    agent_name = args.agent.lower()
    if agent_name not in AGENT_CLASSES:
        raise ValueError(f"Unknown agent type: {agent_name}")
    
    agent_class = AGENT_CLASSES[agent_name]
    
    # Create experiment name if not specified
    experiment_name = args.experiment_name or f"{agent_name}_{config['env_name'].split('/')[-1].lower()}_run1"
    
    # Create trainer and run training
    trainer = RLTrainer(
        agent_class=agent_class,
        config=config,
        experiment_name=experiment_name
    )
    
    results = trainer.train()
    
    return results


if __name__ == "__main__":
    main()