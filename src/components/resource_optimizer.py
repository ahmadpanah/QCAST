
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import asdict
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import time

from ..models.data_models import Container, Metrics, Node
from ..utils.metrics import MetricsCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResourceType:
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    DISK = "disk"
    IO = "io"

class OptimizationStrategy:
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"

class ResourceOptimizer:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Resource Optimizer with configuration
        
        Args:
            config: Configuration dictionary containing:
                - optimization_strategy: Strategy for resource optimization
                - update_interval: Interval for resource updates (seconds)
                - monitoring_window: Window for metrics monitoring (seconds)
                - thresholds: Resource utilization thresholds
                - scaling_factors: Resource scaling factors
                - min_resources: Minimum resource allocations
                - max_resources: Maximum resource allocations
        """
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.resource_history: Dict[str, List[Tuple[datetime, Dict[str, float]]]] = defaultdict(list)
        self.optimization_lock = threading.Lock()
        self.running = True
        
        # Start background optimization thread
        self.optimization_thread = threading.Thread(target=self._background_optimization)
        self.optimization_thread.daemon = True
        self.optimization_thread.start()

    def optimize_resources(self, container: Container, metrics: Metrics, 
                         thresholds: Dict[str, float]) -> Dict[str, float]:
        """
        Implementation of Algorithm 5: Comprehensive Dynamic Resource Optimization
        
        Args:
            container: Container to optimize
            metrics: Current metrics
            thresholds: Resource thresholds
        
        Returns:
            Dict containing optimized resource allocations
        """
        try:
            with self.optimization_lock:
                optimized_resources = {}
                
                # Optimize each resource type
                optimized_resources.update(
                    self._optimize_cpu(container, metrics, thresholds))
                optimized_resources.update(
                    self._optimize_memory(container, metrics, thresholds))
                optimized_resources.update(
                    self._optimize_io(container, metrics, thresholds))
                optimized_resources.update(
                    self._optimize_network(container, metrics, thresholds))
                
                # Detect and resolve resource conflicts
                conflict_metrics = self._detect_resource_conflicts(
                    container, optimized_resources)
                if conflict_metrics['has_conflicts']:
                    optimized_resources = self._resolve_resource_conflicts(
                        container, optimized_resources, conflict_metrics)
                
                # Validate and apply constraints
                optimized_resources = self._apply_resource_constraints(
                    optimized_resources)
                
                # Update resource history
                self._update_resource_history(container.id, optimized_resources)
                
                return optimized_resources
                
        except Exception as e:
            logger.error(f"Error in resource optimization: {str(e)}")
            return container.resources

    def _optimize_cpu(self, container: Container, metrics: Metrics,
                     thresholds: Dict[str, float]) -> Dict[str, float]:
        """Optimize CPU allocation based on usage patterns and performance metrics"""
        try:
            current_cpu = container.resources.get(ResourceType.CPU, 0.0)
            cpu_usage = metrics.cpu_usage
            
            # Calculate base adjustment
            utilization_factor = cpu_usage / thresholds.get('cpu_target', 0.7)
            
            # Consider response time impact
            response_time_factor = metrics.response_time / thresholds.get('response_time', 100)
            
            # Calculate adjustment with weighted factors
            adjustment = self._calculate_resource_adjustment(
                utilization_factor,
                response_time_factor,
                ResourceType.CPU
            )
            
            new_cpu = current_cpu * (1 + adjustment)
            
            # Apply constraints
            new_cpu = max(
                self.config['min_resources']['cpu'],
                min(self.config['max_resources']['cpu'], new_cpu)
            )
            
            return {ResourceType.CPU: new_cpu}
            
        except Exception as e:
            logger.error(f"Error in CPU optimization: {str(e)}")
            return {ResourceType.CPU: container.resources.get(ResourceType.CPU, 0.0)}

    def _optimize_memory(self, container: Container, metrics: Metrics,
                        thresholds: Dict[str, float]) -> Dict[str, float]:
        """Optimize memory allocation based on usage patterns and performance metrics"""
        try:
            current_memory = container.resources.get(ResourceType.MEMORY, 0.0)
            memory_usage = metrics.memory_usage
            
            # Calculate base adjustment
            utilization_factor = memory_usage / thresholds.get('memory_target', 0.7)
            
            # Consider error rate impact
            error_factor = metrics.error_rate / thresholds.get('error_rate', 0.01)
            
            # Calculate adjustment with weighted factors
            adjustment = self._calculate_resource_adjustment(
                utilization_factor,
                error_factor,
                ResourceType.MEMORY
            )
            
            new_memory = current_memory * (1 + adjustment)
            
            # Apply constraints
            new_memory = max(
                self.config['min_resources']['memory'],
                min(self.config['max_resources']['memory'], new_memory)
            )
            
            return {ResourceType.MEMORY: new_memory}
            
        except Exception as e:
            logger.error(f"Error in memory optimization: {str(e)}")
            return {ResourceType.MEMORY: container.resources.get(ResourceType.MEMORY, 0.0)}

    def _optimize_io(self, container: Container, metrics: Metrics,
                    thresholds: Dict[str, float]) -> Dict[str, float]:
        """Optimize I/O allocation based on usage patterns"""
        try:
            current_io = container.resources.get(ResourceType.IO, 0.0)
            io_usage = metrics.disk_io
            
            # Calculate base adjustment
            utilization_factor = io_usage / thresholds.get('io_target', 0.7)
            
            # Calculate adjustment
            adjustment = self._calculate_resource_adjustment(
                utilization_factor,
                1.0,  # No secondary factor for I/O
                ResourceType.IO
            )
            
            new_io = current_io * (1 + adjustment)
            
            # Apply constraints
            new_io = max(
                self.config['min_resources']['io'],
                min(self.config['max_resources']['io'], new_io)
            )
            
            return {ResourceType.IO: new_io}
            
        except Exception as e:
            logger.error(f"Error in I/O optimization: {str(e)}")
            return {ResourceType.IO: container.resources.get(ResourceType.IO, 0.0)}

    def _optimize_network(self, container: Container, metrics: Metrics,
                         thresholds: Dict[str, float]) -> Dict[str, float]:
        """Optimize network allocation based on usage patterns"""
        try:
            current_network = container.resources.get(ResourceType.NETWORK, 0.0)
            network_usage = metrics.network_io
            
            # Calculate base adjustment
            utilization_factor = network_usage / thresholds.get('network_target', 0.7)
            
            # Calculate adjustment
            adjustment = self._calculate_resource_adjustment(
                utilization_factor,
                1.0,  # No secondary factor for network
                ResourceType.NETWORK
            )
            
            new_network = current_network * (1 + adjustment)
            
            # Apply constraints
            new_network = max(
                self.config['min_resources']['network'],
                min(self.config['max_resources']['network'], new_network)
            )
            
            return {ResourceType.NETWORK: new_network}
            
        except Exception as e:
            logger.error(f"Error in network optimization: {str(e)}")
            return {ResourceType.NETWORK: container.resources.get(ResourceType.NETWORK, 0.0)}

    def _detect_resource_conflicts(self, container: Container,
                                 new_resources: Dict[str, float]) -> Dict[str, Any]:
        """Detect conflicts between resource allocations"""
        conflicts = {
            'has_conflicts': False,
            'conflicts': [],
            'severity': 0.0
        }
        
        try:
            # Check CPU-Memory ratio
            cpu_memory_ratio = (new_resources.get(ResourceType.CPU, 0.0) /
                              new_resources.get(ResourceType.MEMORY, 1.0))
            if cpu_memory_ratio > self.config.get('max_cpu_memory_ratio', 1.0):
                conflicts['conflicts'].append({
                    'type': 'cpu_memory_ratio',
                    'severity': cpu_memory_ratio / self.config['max_cpu_memory_ratio']
                })
            
            # Check IO-CPU ratio
            io_cpu_ratio = (new_resources.get(ResourceType.IO, 0.0) /
                          new_resources.get(ResourceType.CPU, 1.0))
            if io_cpu_ratio > self.config.get('max_io_cpu_ratio', 10.0):
                conflicts['conflicts'].append({
                    'type': 'io_cpu_ratio',
                    'severity': io_cpu_ratio / self.config['max_io_cpu_ratio']
                })
            
            conflicts['has_conflicts'] = len(conflicts['conflicts']) > 0
            if conflicts['has_conflicts']:
                conflicts['severity'] = max(c['severity'] for c in conflicts['conflicts'])
            
            return conflicts
            
        except Exception as e:
            logger.error(f"Error in conflict detection: {str(e)}")
            return conflicts

    def _resolve_resource_conflicts(self, container: Container,
                                  resources: Dict[str, float],
                                  conflicts: Dict[str, Any]) -> Dict[str, float]:
        """Resolve detected resource conflicts"""
        try:
            resolved_resources = resources.copy()
            
            for conflict in conflicts['conflicts']:
                if conflict['type'] == 'cpu_memory_ratio':
                    # Adjust CPU or memory to maintain proper ratio
                    if self.config['optimization_strategy'] == OptimizationStrategy.CONSERVATIVE:
                        # Increase memory
                        target_memory = (resolved_resources[ResourceType.CPU] /
                                       self.config['max_cpu_memory_ratio'])
                        resolved_resources[ResourceType.MEMORY] = max(
                            resolved_resources[ResourceType.MEMORY],
                            target_memory
                        )
                    else:
                        # Decrease CPU
                        target_cpu = (resolved_resources[ResourceType.MEMORY] *
                                    self.config['max_cpu_memory_ratio'])
                        resolved_resources[ResourceType.CPU] = min(
                            resolved_resources[ResourceType.CPU],
                            target_cpu
                        )
                
                elif conflict['type'] == 'io_cpu_ratio':
                    # Adjust IO or CPU to maintain proper ratio
                    if self.config['optimization_strategy'] == OptimizationStrategy.CONSERVATIVE:
                        # Increase CPU
                        target_cpu = (resolved_resources[ResourceType.IO] /
                                    self.config['max_io_cpu_ratio'])
                        resolved_resources[ResourceType.CPU] = max(
                            resolved_resources[ResourceType.CPU],
                            target_cpu
                        )
                    else:
                        # Decrease IO
                        target_io = (resolved_resources[ResourceType.CPU] *
                                   self.config['max_io_cpu_ratio'])
                        resolved_resources[ResourceType.IO] = min(
                            resolved_resources[ResourceType.IO],
                            target_io
                        )
            
            return resolved_resources
            
        except Exception as e:
            logger.error(f"Error in conflict resolution: {str(e)}")
            return resources

    def _calculate_resource_adjustment(self, utilization_factor: float,
                                    performance_factor: float,
                                    resource_type: str) -> float:
        """Calculate resource adjustment factor based on utilization and performance"""
        try:
            # Get strategy-specific scaling factors
            strategy = self.config['optimization_strategy']
            base_factor = self.config['scaling_factors'][strategy][resource_type]
            
            # Calculate combined factor
            combined_factor = np.mean([utilization_factor, performance_factor])
            
            # Calculate adjustment with damping
            if combined_factor > 1.0:
                # Scale up
                adjustment = base_factor * (combined_factor - 1.0)
            else:
                # Scale down (more conservative)
                adjustment = base_factor * (combined_factor - 1.0) * 0.5
            
            # Apply damping based on history
            damped_adjustment = self._apply_historical_damping(
                adjustment, resource_type)
            
            return damped_adjustment
            
        except Exception as e:
            logger.error(f"Error in adjustment calculation: {str(e)}")
            return 0.0

    def _apply_historical_damping(self, adjustment: float,
                                resource_type: str) -> float:
        """Apply damping based on historical adjustments"""
        try:
            history_window = timedelta(
                seconds=self.config.get('history_window', 300))
            recent_adjustments = []
            
            for container_id, history in self.resource_history.items():
                # Get recent adjustments for this resource type
                current_time = datetime.now()
                recent = [(t, h[resource_type]) for t, h in history 
                         if current_time - t <= history_window]
                
                if len(recent) >= 2:
                    # Calculate relative changes
                    changes = [(v2 - v1) / v1 
                              for (_, v1), (_, v2) in zip(recent[:-1], recent[1:])]
                    recent_adjustments.extend(changes)
            
            if recent_adjustments:
                # Calculate damping factor based on recent volatility
                volatility = np.std(recent_adjustments)
                damping_factor = 1.0 / (1.0 + volatility)
                return adjustment * damping_factor
            
            return adjustment
            
        except Exception as e:
            logger.error(f"Error in historical damping: {str(e)}")
            return adjustment

    def _apply_resource_constraints(self, resources: Dict[str, float]) -> Dict[str, float]:
        """Apply minimum and maximum resource constraints"""
        constrained_resources = {}
        
        for resource_type, value in resources.items():
            min_value = self.config['min_resources'][resource_type]
            max_value = self.config['max_resources'][resource_type]
            constrained_resources[resource_type] = max(min_value, min(max_value, value))
        
        return constrained_resources

    def _update_resource_history(self, container_id: str,
                               resources: Dict[str, float]):
        """Update resource allocation history"""
        try:
            current_time = datetime.now()
            history_max_size = self.config.get('history_max_size', 1000)
            
            self.resource_history[container_id].append((current_time, resources))
            
            # Trim history if needed
            if len(self.resource_history[container_id]) > history_max_size:
                self.resource_history[container_id] = (
                    self.resource_history[container_id][-history_max_size:]
                )
                
        except Exception as e:
            logger.error(f"Error updating resource history: {str(e)}")

    def _background_optimization(self):
        """Background thread for periodic optimization"""
        while self.running:
            try:
                time.sleep(self.config.get('update_interval', 60))
                
                # Collect current metrics
                metrics = self.metrics_collector.collect_system_metrics()
                
                # Perform optimization for all containers
                for container_id, container in self.containers.items():
                    optimized_resources = self.optimize_resources(
                        container, metrics, self.config['thresholds']
                    )
                    
                    # Apply optimized resources
                    self._apply_resources(container_id, optimized_resources)
                
            except Exception as e:
                logger.error(f"Error in background optimization: {str(e)}")

    def _apply_resources(self, container_id: str, resources: Dict[str, float]):
        """Apply resource updates to container"""
        try:
            # Implementation would depend on container runtime
            logger.info(f"Applying resources to container {container_id}: {resources}")
            
        except Exception as e:
            logger.error(f"Error applying resources: {str(e)}")

    def cleanup(self):
        """Cleanup resources and stop background thread"""
        self.running = False
        if self.optimization_thread.is_alive():
            self.optimization_thread.join()

# Example usage
def example_usage():
    # Configuration
    config = {
        'optimization_strategy': OptimizationStrategy.BALANCED,
        'update_interval': 60,
        'monitoring_window': 300,
        'history_window': 300,
        'history_max_size': 1000,
        'thresholds': {
            'cpu_target': 0.7,
            'memory_target': 0.7,
            'io_target': 0.7,
            'network_target': 0.7,
            'response_time': 100,
            'error_rate': 0.01
        },
        'scaling_factors': {
            OptimizationStrategy.CONSERVATIVE: {
                ResourceType.CPU: 0.1,
                ResourceType.MEMORY: 0.1,
                ResourceType.IO: 0.1,
                ResourceType.NETWORK: 0.1
            },
            OptimizationStrategy.BALANCED: {
                ResourceType.CPU: 0.2,
                ResourceType.MEMORY: 0.2,
                ResourceType.IO: 0.2,
                ResourceType.NETWORK: 0.2
            },
            OptimizationStrategy.AGGRESSIVE: {
                ResourceType.CPU: 0.3,
                ResourceType.MEMORY: 0.3,
                ResourceType.IO: 0.3,
                ResourceType.NETWORK: 0.3
            }
        },
        'min_resources': {
            ResourceType.CPU: 0.1,
            ResourceType.MEMORY: 128,
            ResourceType.IO: 10,
            ResourceType.NETWORK: 10
        },
        'max_resources': {
            ResourceType.CPU: 4.0,
            ResourceType.MEMORY: 8192,
            ResourceType.IO: 1000,
            ResourceType.NETWORK: 1000
        },
        'max_cpu_memory_ratio': 1.0,
        'max_io_cpu_ratio': 10.0
    }

    # Initialize optimizer
    optimizer = ResourceOptimizer(config)

    # Example container and metrics
    container = Container(
        id="container1",
        status="running",
        resources={
            ResourceType.CPU: 1.0,
            ResourceType.MEMORY: 1024,
            ResourceType.IO: 100,
            ResourceType.NETWORK: 100
        },
        metrics=Metrics(
            cpu_usage=0.8,
            memory_usage=0.7,
            network_io=50,
            disk_io=40,
            request_count=100,
            response_time=150,
            error_rate=0.02
        ),
        creation_time=time.time()
    )

    # Optimize resources
    optimized_resources = optimizer.optimize_resources(
        container,
        container.metrics,
        config['thresholds']
    )

    print("Optimized resources:", optimized_resources)

    # Cleanup
    optimizer.cleanup()

if __name__ == "__main__":
    example_usage()