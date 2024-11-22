
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from collections import defaultdict
import random
import json
import logging
from datetime import datetime
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class State:
    """Class representing the system state"""
    def __init__(self, metrics: Dict[str, float], containers: int, demand: int):
        self.metrics = metrics
        self.containers = containers
        self.demand = demand
    
    def to_tuple(self) -> tuple:
        """Convert state to tuple for Q-table lookup"""
        return (
            self._discretize_value(self.metrics.get('cpu_usage', 0), 10),
            self._discretize_value(self.metrics.get('memory_usage', 0), 10),
            self._discretize_value(self.metrics.get('response_time', 0), 10),
            self._discretize_value(self.metrics.get('error_rate', 0), 10),
            self._discretize_value(self.containers / 100, 10),  # Normalize containers
            self._discretize_value(self.demand / 1000, 10)      # Normalize demand
        )
    
    @staticmethod
    def _discretize_value(value: float, bins: int) -> int:
        """Discretize continuous values into bins"""
        return min(bins - 1, int(value * bins))

class Action:
    """Available actions for the Q-learning agent"""
    SCALE_UP_LARGE = "scale_up_large"    # Scale up by 5 containers
    SCALE_UP_SMALL = "scale_up_small"    # Scale up by 1 container
    MAINTAIN = "maintain"                 # Maintain current state
    SCALE_DOWN_SMALL = "scale_down_small"  # Scale down by 1 container
    SCALE_DOWN_LARGE = "scale_down_large"  # Scale down by 5 containers
    
    @classmethod
    def get_all_actions(cls) -> List[str]:
        return [cls.SCALE_UP_LARGE, cls.SCALE_UP_SMALL, cls.MAINTAIN, 
                cls.SCALE_DOWN_SMALL, cls.SCALE_DOWN_LARGE]

class ExperienceBuffer:
    """Experience replay buffer for improved learning"""
    def __init__(self, max_size: int = 10000):
        self.buffer: List[Tuple] = []
        self.max_size = max_size
    
    def add(self, state: tuple, action: str, reward: float, next_state: tuple):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

class QLearningEngine:
    """Q-Learning Engine with advanced features and optimizations"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Q-Learning Engine with configuration
        
        Args:
            config: Configuration dictionary containing:
                - learning_rate: Initial learning rate
                - discount_factor: Future reward discount factor
                - initial_epsilon: Initial exploration rate
                - min_epsilon: Minimum exploration rate
                - epsilon_decay: Epsilon decay rate
                - batch_size: Experience replay batch size
                - save_interval: Interval to save Q-table
                - model_path: Path to save/load model
        """
        self.config = config
        self.learning_rate = config.get('learning_rate', 0.1)
        self.discount_factor = config.get('discount_factor', 0.95)
        self.epsilon = config.get('initial_epsilon', 1.0)
        self.min_epsilon = config.get('min_epsilon', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.batch_size = config.get('batch_size', 32)
        
        # Initialize Q-table and experience buffer
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.experience_buffer = ExperienceBuffer(config.get('buffer_size', 10000))
        
        # Learning statistics
        self.training_stats = {
            'episodes': 0,
            'total_rewards': 0,
            'avg_rewards': [],
            'epsilon_history': [],
            'loss_history': []
        }
        
        # Load existing model if available
        self.model_path = Path(config.get('model_path', 'models/q_learning_model.pkl'))
        if self.model_path.exists():
            self.load_model()

    def update_q_value(self, state: tuple, action: str, reward: float, 
                      next_state: tuple) -> float:
        """
        Implementation of Algorithm 1: Enhanced Q-Learning State-Action Update
        
        Args:
            state: Current state tuple
            action: Selected action
            reward: Received reward
            next_state: Next state tuple
        
        Returns:
            float: Updated Q-value
        """
        try:
            # Add experience to buffer
            self.experience_buffer.add(state, action, reward, next_state)
            
            # Perform experience replay
            if len(self.experience_buffer.buffer) >= self.batch_size:
                return self._experience_replay()
            
            # Regular Q-value update
            return self._update_single_q_value(state, action, reward, next_state)
            
        except Exception as e:
            logger.error(f"Error in Q-value update: {str(e)}")
            return self.q_table[state][action]

    def select_action(self, state: tuple, valid_actions: List[str]) -> str:
        """
        Implementation of Algorithm 2: Action Selection with Adaptive ϵ-greedy Strategy
        
        Args:
            state: Current state tuple
            valid_actions: List of valid actions
        
        Returns:
            str: Selected action
        """
        try:
            # Update exploration rate
            self._update_epsilon()
            
            # Adapt epsilon based on state uncertainty
            adapted_epsilon = self._adapt_epsilon(state)
            
            # Exploration
            if random.random() < adapted_epsilon:
                return self._prioritized_random_action(valid_actions, state)
            
            # Exploitation
            return self._get_best_action(state, valid_actions)
            
        except Exception as e:
            logger.error(f"Error in action selection: {str(e)}")
            return random.choice(valid_actions)

    def _update_single_q_value(self, state: tuple, action: str, reward: float, 
                             next_state: tuple) -> float:
        """Update single Q-value using standard Q-learning update rule"""
        try:
            current_value = self.q_table[state][action]
            next_max_value = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
            
            # Calculate convergence metric and adapt learning rate
            convergence_metric = self._calculate_convergence()
            adapted_learning_rate = self._adapt_learning_rate(self.learning_rate, 
                                                            convergence_metric)
            
            # Update Q-value
            new_value = current_value + adapted_learning_rate * (
                reward + self.discount_factor * next_max_value - current_value
            )
            
            self.q_table[state][action] = new_value
            self._update_training_stats(reward)
            
            return new_value
            
        except Exception as e:
            logger.error(f"Error in single Q-value update: {str(e)}")
            return current_value

    def _experience_replay(self) -> float:
        """Perform experience replay update"""
        try:
            total_loss = 0
            experiences = self.experience_buffer.sample(self.batch_size)
            
            for state, action, reward, next_state in experiences:
                new_value = self._update_single_q_value(state, action, reward, next_state)
                total_loss += abs(new_value - self.q_table[state][action])
            
            avg_loss = total_loss / len(experiences)
            self.training_stats['loss_history'].append(avg_loss)
            
            return avg_loss
            
        except Exception as e:
            logger.error(f"Error in experience replay: {str(e)}")
            return 0.0

    def _adapt_epsilon(self, state: tuple) -> float:
        """Adapt epsilon based on state uncertainty"""
        try:
            # Calculate state uncertainty
            state_values = list(self.q_table[state].values())
            if not state_values:
                return self.epsilon
            
            # Normalize uncertainty between 0 and 1
            uncertainty = np.std(state_values) / (np.max(state_values) - np.min(state_values) + 1e-6)
            
            # Increase epsilon for high uncertainty states
            return min(1.0, self.epsilon * (1 + uncertainty))
            
        except Exception as e:
            logger.error(f"Error in epsilon adaptation: {str(e)}")
            return self.epsilon

    def _prioritized_random_action(self, valid_actions: List[str], 
                                 state: tuple) -> str:
        """Select random action with prioritization"""
        try:
            # Calculate action priorities based on historical performance
            action_priorities = {}
            for action in valid_actions:
                # Combine historical Q-values with exploration bonus
                q_value = self.q_table[state][action]
                visit_count = sum(1 for s in self.q_table if action in self.q_table[s])
                exploration_bonus = 1.0 / (visit_count + 1)
                
                action_priorities[action] = q_value + exploration_bonus
            
            # Select action using softmax distribution
            priorities = np.array(list(action_priorities.values()))
            probabilities = self._softmax(priorities)
            
            return np.random.choice(valid_actions, p=probabilities)
            
        except Exception as e:
            logger.error(f"Error in prioritized random action: {str(e)}")
            return random.choice(valid_actions)

    def _get_best_action(self, state: tuple, valid_actions: List[str]) -> str:
        """Get best action based on Q-values"""
        try:
            if not self.q_table[state]:
                return random.choice(valid_actions)
            
            return max(valid_actions, key=lambda a: self.q_table[state].get(a, 0))
            
        except Exception as e:
            logger.error(f"Error in getting best action: {str(e)}")
            return random.choice(valid_actions)

    def _calculate_convergence(self) -> float:
        """Calculate convergence metric based on Q-value stability"""
        try:
            if len(self.training_stats['loss_history']) < 2:
                return 0.5
            
            recent_losses = self.training_stats['loss_history'][-10:]
            if not recent_losses:
                return 0.5
            
            # Calculate trend in recent losses
            loss_trend = np.mean(np.diff(recent_losses))
            
            # Normalize trend to [0, 1] range
            convergence = 1.0 / (1.0 + np.exp(loss_trend))
            
            return convergence
            
        except Exception as e:
            logger.error(f"Error in convergence calculation: {str(e)}")
            return 0.5

    def _adapt_learning_rate(self, base_rate: float, 
                           convergence_metric: float) -> float:
        """Adapt learning rate based on convergence"""
        try:
            # Reduce learning rate as system converges
            return base_rate * (1 - convergence_metric)
            
        except Exception as e:
            logger.error(f"Error in learning rate adaptation: {str(e)}")
            return base_rate

    def _update_epsilon(self):
        """Update exploration rate"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.training_stats['epsilon_history'].append(self.epsilon)

    def _update_training_stats(self, reward: float):
        """Update training statistics"""
        self.training_stats['episodes'] += 1
        self.training_stats['total_rewards'] += reward
        
        avg_reward = self.training_stats['total_rewards'] / self.training_stats['episodes']
        self.training_stats['avg_rewards'].append(avg_reward)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax values for each set of scores in x"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def save_model(self):
        """Save Q-learning model and training stats"""
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'q_table': dict(self.q_table),
                'training_stats': self.training_stats,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved successfully to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def load_model(self):
        """Load Q-learning model and training stats"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_table = defaultdict(lambda: defaultdict(float), model_data['q_table'])
            self.training_stats = model_data['training_stats']
            self.config.update(model_data['config'])
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'episodes': self.training_stats['episodes'],
            'total_rewards': self.training_stats['total_rewards'],
            'average_reward': np.mean(self.training_stats['avg_rewards'][-100:]),
            'current_epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
            'experience_buffer_size': len(self.experience_buffer.buffer)
        }

# Usage example
def example_usage():
    # Configuration
    config = {
        'learning_rate': 0.1,
        'discount_factor': 0.95,
        'initial_epsilon': 1.0,
        'min_epsilon': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 32,
        'buffer_size': 10000,
        'model_path': 'models/q_learning_model.pkl'
    }
    
    # Initialize Q-learning engine
    q_learning = QLearningEngine(config)
    
    # Example state and actions
    current_state = State(
        metrics={
            'cpu_usage': 0.7,
            'memory_usage': 0.6,
            'response_time': 100,
            'error_rate': 0.01
        },
        containers=10,
        demand=100
    ).to_tuple()
    
    valid_actions = Action.get_all_actions()
    
    # Select action
    action = q_learning.select_action(current_state, valid_actions)
    
    # Simulate environment interaction
    reward = 1.0  # Example reward
    next_state = State(
        metrics={
            'cpu_usage': 0.65,
            'memory_usage': 0.55,
            'response_time': 90,
            'error_rate': 0.009
        },
        containers=11,
        demand=100
    ).to_tuple()
    
    # Update Q-values
    q_learning.update_q_value(current_state, action, reward, next_state)
    
    # Get statistics
    stats = q_learning.get_statistics()
    print("Training Statistics:", stats)
    
    # Save model
    q_learning.save_model()

if __name__ == "__main__":
    example_usage()