# src/core/qcast_system.py
from typing import Dict, List, Any, Tuple, Optional
import threading
import time
import logging
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from dataclasses import asdict

from ..components.q_learning_engine import QLearningEngine, State, Action
from ..components.container_lifecycle_manager import ContainerLifecycleManager
from ..components.resource_optimizer import ResourceOptimizer
from ..components.workload_monitor import WorkloadMonitor
from ..models.data_models import Container, Metrics, Node
from ..utils.metrics import MetricsCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemState:
    """Enumeration of system states"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"

class QCASTSystem:
    def __init__(self, config_path: str):
        """
        Initialize QCAST System with configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.system_state = SystemState.INITIALIZING
        
        # Initialize components
        self.q_learning_engine = QLearningEngine(self.config['q_learning'])
        self.container_manager = ContainerLifecycleManager(self.config['container'])
        self.resource_optimizer = ResourceOptimizer(self.config['resources'])
        self.workload_monitor = WorkloadMonitor(self.config['monitoring'])
        
        # System metrics
        self.metrics_collector = MetricsCollector()
        self.system_metrics: Dict[str, Any] = {}
        self.performance_history: List[Dict[str, float]] = []
        
        # Control flags and locks
        self.running = True
        self.maintenance_mode = False
        self.control_lock = threading.Lock()
        
        # Initialize monitoring and control threads
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.control_thread = threading.Thread(target=self._control_loop)
        self.optimization_thread = threading.Thread(target=self._optimization_loop)
        
        # Start threads
        self._start_system()

    def run(self, monitoring_interval: float = 1.0):
        """
        Implementation of Algorithm 7: QCAST Main Control Loop
        
        Args:
            monitoring_interval: Interval between monitoring cycles
        """
        try:
            logger.info("Starting QCAST system")
            self.system_state = SystemState.RUNNING
            
            while self.running:
                with self.control_lock:
                    try:
                        # Collect current metrics
                        metrics = self.metrics_collector.collect_system_metrics()
                        self.system_metrics = asdict(metrics)
                        
                        # Construct current state
                        current_state = self._construct_state(metrics)
                        
                        # Select action using Q-learning
                        valid_actions = self._get_valid_actions()
                        action = self.q_learning_engine.select_action(
                            current_state, valid_actions)
                        
                        # Execute action
                        self._execute_action(action)
                        
                        # Collect new metrics and calculate reward
                        new_metrics = self.metrics_collector.collect_system_metrics()
                        new_state = self._construct_state(new_metrics)
                        reward = self._calculate_reward(new_metrics)
                        
                        # Update Q-learning
                        self.q_learning_engine.update_q_value(
                            current_state, action, reward, new_state)
                        
                        # Update performance history
                        self._update_performance_history(metrics, action, reward)
                        
                        # Check for maintenance needs
                        if self._needs_maintenance():
                            self._enter_maintenance_mode()
                        
                    except Exception as e:
                        logger.error(f"Error in main control loop: {str(e)}")
                        self.system_state = SystemState.ERROR
                        self._handle_error()
                
                time.sleep(monitoring_interval)
                
        except Exception as e:
            logger.error(f"Critical error in QCAST system: {str(e)}")
            self.cleanup()

    def _start_system(self):
        """Initialize and start system components and threads"""
        try:
            # Start monitoring thread
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            # Start control thread
            self.control_thread.daemon = True
            self.control_thread.start()
            
            # Start optimization thread
            self.optimization_thread.daemon = True
            self.optimization_thread.start()
            
            logger.info("All system threads started successfully")
            
        except Exception as e:
            logger.error(f"Error starting system: {str(e)}")
            self.cleanup()

    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                # Collect and analyze metrics
                metrics = self.metrics_collector.collect_system_metrics()
                self.workload_monitor.add_metrics(metrics)
                
                # Analyze workload patterns
                analysis = self.workload_monitor.analyze_workload(
                    list(self.workload_monitor.history),
                    self.config['monitoring']['window_size']
                )
                
                # Update system metrics
                self.system_metrics.update({
                    'workload_patterns': analysis['patterns'],
                    'anomalies': analysis['anomalies'],
                    'predictions': analysis['predictions']
                })
                
                time.sleep(self.config['monitoring']['interval'])
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")

    def _control_loop(self):
        """Background control loop"""
        while self.running:
            try:
                if self.system_state == SystemState.RUNNING:
                    # Implement control policies
                    self._apply_control_policies()
                    
                    # Check system health
                    self._check_system_health()
                    
                    # Update resource allocations
                    self._update_resource_allocations()
                
                time.sleep(self.config['control']['interval'])
                
            except Exception as e:
                logger.error(f"Error in control loop: {str(e)}")

    def _optimization_loop(self):
        """Background optimization loop"""
        while self.running:
            try:
                if self.system_state == SystemState.RUNNING:
                    # Optimize container placement
                    self._optimize_container_placement()
                    
                    # Optimize resource allocation
                    self._optimize_resource_allocation()
                    
                    # Optimize scaling decisions
                    self._optimize_scaling_decisions()
                
                time.sleep(self.config['optimization']['interval'])
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {str(e)}")

    def _construct_state(self, metrics: Metrics) -> tuple:
        """Construct system state representation"""
        try:
            # Create state object with current metrics and system info
            state = State(
                metrics=asdict(metrics),
                containers=len(self.container_manager.get_active_containers()),
                demand=self.system_metrics.get('request_count', 0)
            )
            
            return state.to_tuple()
            
        except Exception as e:
            logger.error(f"Error constructing state: {str(e)}")
            return tuple()

    def _get_valid_actions(self) -> List[str]:
        """Get list of valid actions based on current state"""
        try:
            valid_actions = []
            
            # Check scaling constraints
            current_containers = len(self.container_manager.get_active_containers())
            
            if current_containers < self.config['container']['max_containers']:
                valid_actions.extend([Action.SCALE_UP_SMALL, Action.SCALE_UP_LARGE])
            
            if current_containers > self.config['container']['min_containers']:
                valid_actions.extend([Action.SCALE_DOWN_SMALL, Action.SCALE_DOWN_LARGE])
            
            valid_actions.append(Action.MAINTAIN)
            
            return valid_actions
            
        except Exception as e:
            logger.error(f"Error getting valid actions: {str(e)}")
            return [Action.MAINTAIN]

    def _execute_action(self, action: str):
        """Execute selected action"""
        try:
            logger.info(f"Executing action: {action}")
            
            if action == Action.SCALE_UP_LARGE:
                self.container_manager.execute_action('scale_up', {'count': 5})
            elif action == Action.SCALE_UP_SMALL:
                self.container_manager.execute_action('scale_up', {'count': 1})
            elif action == Action.SCALE_DOWN_LARGE:
                self.container_manager.execute_action('scale_down', {'count': 5})
            elif action == Action.SCALE_DOWN_SMALL:
                self.container_manager.execute_action('scale_down', {'count': 1})
            elif action == Action.MAINTAIN:
                self.container_manager.execute_action('maintain')
            
        except Exception as e:
            logger.error(f"Error executing action {action}: {str(e)}")

    def _calculate_reward(self, metrics: Metrics) -> float:
        """Calculate reward based on system metrics"""
        try:
            # Get target values from config
            targets = self.config['rewards']['targets']
            weights = self.config['rewards']['weights']
            
            # Calculate individual rewards
            rewards = {
                'cpu': self._calculate_resource_reward(
                    metrics.cpu_usage, 
                    targets['cpu_usage'],
                    weights['cpu']
                ),
                'memory': self._calculate_resource_reward(
                    metrics.memory_usage,
                    targets['memory_usage'],
                    weights['memory']
                ),
                'response_time': self._calculate_performance_reward(
                    metrics.response_time,
                    targets['response_time'],
                    weights['response_time']
                ),
                'error_rate': self._calculate_performance_reward(
                    metrics.error_rate,
                    targets['error_rate'],
                    weights['error_rate']
                )
            }
            
            # Calculate total reward
            total_reward = sum(rewards.values())
            
            return total_reward
            
        except Exception as e:
            logger.error(f"Error calculating reward: {str(e)}")
            return 0.0

    def _calculate_resource_reward(self, value: float, target: float, 
                                 weight: float) -> float:
        """Calculate reward for resource metrics"""
        try:
            # Penalize both under and over utilization
            deviation = abs(value - target)
            reward = weight * (1 - deviation)
            return max(-1.0, min(1.0, reward))
            
        except Exception as e:
            logger.error(f"Error calculating resource reward: {str(e)}")
            return 0.0

    def _calculate_performance_reward(self, value: float, target: float,
                                    weight: float) -> float:
        """Calculate reward for performance metrics"""
        try:
            # Penalize only when worse than target
            if value <= target:
                return weight
            
            deviation = (value - target) / target
            reward = weight * (1 - deviation)
            return max(-1.0, min(1.0, reward))
            
        except Exception as e:
            logger.error(f"Error calculating performance reward: {str(e)}")
            return 0.0

    def _update_performance_history(self, metrics: Metrics, action: str,
                                  reward: float):
        """Update performance history"""
        try:
            performance_entry = {
                'timestamp': datetime.now().isoformat(),
                'metrics': asdict(metrics),
                'action': action,
                'reward': reward
            }
            
            self.performance_history.append(performance_entry)
            
            # Trim history if needed
            max_history = self.config['system']['max_history_size']
            if len(self.performance_history) > max_history:
                self.performance_history = self.performance_history[-max_history:]
                
        except Exception as e:
            logger.error(f"Error updating performance history: {str(e)}")

    def _needs_maintenance(self) -> bool:
        """Check if system needs maintenance"""
        try:
            # Check system health indicators
            health_checks = [
                self._check_resource_health(),
                self._check_performance_health(),
                self._check_container_health()
            ]
            
            return any(not check for check in health_checks)
            
        except Exception as e:
            logger.error(f"Error checking maintenance needs: {str(e)}")
            return False

    def _enter_maintenance_mode(self):
        """Enter system maintenance mode"""
        try:
            logger.info("Entering maintenance mode")
            self.system_state = SystemState.MAINTENANCE
            self.maintenance_mode = True
            
            # Perform maintenance tasks
            self._perform_maintenance()
            
            # Exit maintenance mode
            self.maintenance_mode = False
            self.system_state = SystemState.RUNNING
            logger.info("Exiting maintenance mode")
            
        except Exception as e:
            logger.error(f"Error in maintenance mode: {str(e)}")
            self.system_state = SystemState.ERROR

    def _perform_maintenance(self):
        """Perform system maintenance tasks"""
        try:
            # Save system state
            self._save_system_state()
            
            # Cleanup resources
            self._cleanup_resources()
            
            # Optimize components
            self._optimize_components()
            
            # Verify system health
            if not self._verify_system_health():
                raise Exception("System health verification failed")
                
        except Exception as e:
            logger.error(f"Error performing maintenance: {str(e)}")
            self._handle_error()

    def _handle_error(self):
        """Handle system errors"""
        try:
            logger.error("Handling system error")
            
            # Save error state
            self._save_error_state()
            
            # Attempt recovery
            recovery_successful = self._attempt_recovery()
            
            if recovery_successful:
                self.system_state = SystemState.RUNNING
                logger.info("System recovered successfully")
            else:
                logger.error("System recovery failed")
                self.cleanup()
                
        except Exception as e:
            logger.error(f"Error in error handler: {str(e)}")
            self.cleanup()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load system configuration"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Validate configuration
            self._validate_config(config)
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise

    def _validate_config(self, config: Dict[str, Any]):
        """Validate system configuration"""
        required_sections = ['q_learning', 'container', 'resources', 'monitoring',
                           'control', 'optimization', 'rewards', 'system']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing configuration section: {section}")

    def _save_system_state(self):
        """Save current system state"""
        try:
            state_path = Path(self.config['system']['state_path'])
            state_path.parent.mkdir(parents=True, exist_ok=True)
            
            state_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': self.system_metrics,
                'performance_history': self.performance_history,
                'system_state': self.system_state
            }
            
            with open(state_path, 'w') as f:
                json.dump(state_data, f)
                
        except Exception as e:
            logger.error(f"Error saving system state: {str(e)}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'state': self.system_state,
            'metrics': self.system_metrics,
            'containers': len(self.container_manager.get_active_containers()),
            'maintenance_mode': self.maintenance_mode,
            'timestamp': datetime.now().isoformat()
        }

    def cleanup(self):
        """Cleanup system resources"""
        try:
            logger.info("Cleaning up QCAST system")
            self.running = False
            self.system_state = SystemState.SHUTTING_DOWN
            
            # Stop components
            self.workload_monitor.cleanup()
            self.resource_optimizer.cleanup()
            self.container_manager.cleanup()
            
            # Save final state
            self._save_system_state()
            
            # Wait for threads to finish
            self.monitor_thread.join()
            self.control_thread.join()
            self.optimization_thread.join()
            
            logger.info("QCAST system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error in system cleanup: {str(e)}")

# Example usage
def main():
    # Load system configuration
    config_path = "config/qcast_config.json"
    
    try:
        # Initialize and start QCAST system
        qcast = QCASTSystem(config_path)
        qcast.run()
        
    except Exception as e:
        logger.error(f"Error in QCAST system: {str(e)}")
        
    finally:
        qcast.cleanup()

if __name__ == "__main__":
    main()