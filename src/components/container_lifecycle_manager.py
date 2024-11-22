# src/components/container_lifecycle_manager.py
from typing import Dict, List, Set, Tuple, Optional, Any
from datetime import datetime, timedelta
import uuid
import logging
import numpy as np
from dataclasses import asdict
from collections import defaultdict

from ..models.data_models import Container, Node, Metrics, DemandForecast
from ..components.demand_predictor import DemandPredictor
from ..components.optimal_size_computer import OptimalSizeComputer
from ..utils.metrics import MetricsCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContainerState:
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    IDLE = "idle"
    TERMINATING = "terminating"
    TERMINATED = "terminated"

class ContainerLifecycleManager:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Container Lifecycle Manager with configuration.
        
        Args:
            config: Configuration dictionary containing:
                - min_containers: Minimum number of containers
                - max_containers: Maximum number of containers
                - container_capacity: Requests per container
                - warmup_time: Container warmup time in seconds
                - idle_timeout: Container idle timeout in seconds
                - resource_limits: Dict of resource limits per container
        """
        self.config = config
        self.containers: Dict[str, Container] = {}
        self.nodes: Dict[str, Node] = {}
        self.container_states: Dict[str, str] = {}
        self.container_metrics: Dict[str, MetricsCollector] = {}
        self.demand_predictor = DemandPredictor()
        self.optimal_size_computer = OptimalSizeComputer(config)
        
        # Container pools
        self.warm_pool: Set[str] = set()
        self.cold_pool: Set[str] = set()
        
        # Container lifecycle tracking
        self.container_last_used: Dict[str, datetime] = {}
        self.container_creation_times: Dict[str, datetime] = {}
        
        # Performance tracking
        self.performance_history: List[Dict[str, float]] = []

    def manage_container_pool(self, current_demand: int, prediction_window: int, 
                            resource_constraints: Dict[str, float]) -> None:
        """
        Implementation of Algorithm 3: Proactive Container Pool Management
        
        Args:
            current_demand: Current request demand
            prediction_window: Time window for demand prediction in seconds
            resource_constraints: Available resources for containers
        """
        try:
            # Get current pool status
            current_pool = self.get_active_containers()
            current_size = len(current_pool)
            
            # Predict future demand
            demand_forecast = self.demand_predictor.predict_demand(prediction_window)
            target_size = self.optimal_size_computer.compute_optimal_size(
                demand_forecast, resource_constraints, self._get_current_metrics()
            )
            
            # Assess pool health and rebalance if needed
            health_metrics = self._assess_pool_health(current_pool)
            if health_metrics['requires_rebalancing']:
                self._rebalance_pool(current_pool)
            
            # Scale pool size
            if current_size < target_size:
                containers_needed = target_size - current_size
                self._scale_up(containers_needed, resource_constraints)
            elif current_size > target_size:
                containers_excess = current_size - target_size
                self._scale_down(containers_excess)
            
            # Maintain warm pool
            self._maintain_warm_pool(demand_forecast)
            
            # Cleanup idle containers
            self._cleanup_idle_containers()
            
            logger.info(f"Pool management completed. Current size: {current_size}, "
                       f"Target size: {target_size}")
            
        except Exception as e:
            logger.error(f"Error in container pool management: {str(e)}")
            raise

    def optimize_placement(self, containers: List[Container], nodes: List[Node], 
                         resource_requirements: Dict[str, float]) -> Dict[str, str]:
        """
        Implementation of Algorithm 4: Container Placement Optimization
        
        Args:
            containers: List of containers to place
            nodes: List of available nodes
            resource_requirements: Resource requirements per container
        
        Returns:
            Dict mapping container IDs to node IDs
        """
        placement_map = {}
        
        try:
            # Sort containers by priority
            sorted_containers = self._sort_containers_by_priority(containers)
            
            # Track remaining resources per node
            remaining_resources = {
                node.id: node.available_resources.copy() for node in nodes
            }
            
            for container in sorted_containers:
                best_node = None
                min_cost = float('inf')
                
                # Find best node for container
                for node in nodes:
                    if self._can_accommodate(node, container, resource_requirements,
                                          remaining_resources[node.id]):
                        cost = self._calculate_placement_cost(container, node,
                                                           remaining_resources[node.id])
                        if cost < min_cost:
                            min_cost = cost
                            best_node = node
                
                if best_node:
                    # Update placement and resources
                    placement_map[container.id] = best_node.id
                    self._update_node_resources(best_node, container,
                                             remaining_resources[best_node.id])
                else:
                    logger.warning(f"No suitable node found for container {container.id}")
            
            logger.info(f"Placement optimization completed for {len(containers)} containers")
            
        except Exception as e:
            logger.error(f"Error in container placement optimization: {str(e)}")
            raise
        
        return placement_map

    def execute_action(self, action: str, params: Dict[str, Any] = None) -> bool:
        """
        Execute container management action
        
        Args:
            action: Action to execute (scale_up, scale_down, rebalance, etc.)
            params: Additional parameters for the action
        
        Returns:
            bool: Success status
        """
        try:
            if action == "scale_up":
                return self._scale_up(params.get("count", 1), params.get("resources", {}))
            elif action == "scale_down":
                return self._scale_down(params.get("count", 1))
            elif action == "rebalance":
                return self._rebalance_pool(self.get_active_containers())
            else:
                logger.warning(f"Unknown action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing action {action}: {str(e)}")
            return False

    def _scale_up(self, count: int, resource_constraints: Dict[str, float]) -> bool:
        """Scale up the container pool by creating new containers"""
        try:
            for _ in range(count):
                container_id = str(uuid.uuid4())
                container = Container(
                    id=container_id,
                    status=ContainerState.PENDING,
                    resources=self.config['resource_limits'].copy(),
                    metrics=self._create_empty_metrics(),
                    creation_time=datetime.now().timestamp()
                )
                
                self.containers[container_id] = container
                self.container_states[container_id] = ContainerState.PENDING
                self.container_creation_times[container_id] = datetime.now()
                
                # Initialize container metrics collector
                self.container_metrics[container_id] = MetricsCollector()
                
                # Start container initialization
                success = self._initialize_container(container_id)
                if not success:
                    logger.error(f"Failed to initialize container {container_id}")
                    self._cleanup_container(container_id)
                    continue
                
                logger.info(f"Container {container_id} created and initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in scale up operation: {str(e)}")
            return False

    def _scale_down(self, count: int) -> bool:
        """Scale down the container pool by removing containers"""
        try:
            containers_to_remove = self._select_containers_for_removal(count)
            
            for container_id in containers_to_remove:
                # Gracefully shutdown container
                success = self._graceful_shutdown(container_id)
                if not success:
                    logger.warning(f"Graceful shutdown failed for container {container_id}")
                    # Force removal if graceful shutdown fails
                    self._force_remove_container(container_id)
                
                # Clean up container resources
                self._cleanup_container(container_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in scale down operation: {str(e)}")
            return False

    def _rebalance_pool(self, current_pool: List[Container]) -> bool:
        """Rebalance container pool to optimize resource distribution"""
        try:
            # Get current resource distribution
            node_resources = self._get_node_resource_distribution()
            
            # Calculate optimal distribution
            optimal_distribution = self._calculate_optimal_distribution(
                current_pool, self.nodes.values()
            )
            
            # Perform container migrations
            migrations = self._plan_migrations(
                current_pool, node_resources, optimal_distribution
            )
            
            for container_id, target_node_id in migrations.items():
                success = self._migrate_container(container_id, target_node_id)
                if not success:
                    logger.warning(f"Migration failed for container {container_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in pool rebalancing: {str(e)}")
            return False

    def _assess_pool_health(self, pool: List[Container]) -> Dict[str, Any]:
        """
        Assess the health of the container pool
        
        Returns:
            Dict containing health metrics and rebalancing requirement
        """
        health_metrics = {
            'requires_rebalancing': False,
            'unhealthy_containers': [],
            'resource_imbalance': 0.0,
            'performance_issues': []
        }
        
        try:
            # Check container health
            for container in pool:
                container_health = self._check_container_health(container)
                if not container_health['healthy']:
                    health_metrics['unhealthy_containers'].append(container.id)
            
            # Check resource distribution
            resource_distribution = self._get_node_resource_distribution()
            resource_imbalance = self._calculate_resource_imbalance(resource_distribution)
            health_metrics['resource_imbalance'] = resource_imbalance
            
            # Check performance metrics
            performance_issues = self._check_performance_metrics(pool)
            health_metrics['performance_issues'] = performance_issues
            
            # Determine if rebalancing is needed
            health_metrics['requires_rebalancing'] = (
                len(health_metrics['unhealthy_containers']) > 0 or
                resource_imbalance > self.config.get('max_resource_imbalance', 0.2) or
                len(performance_issues) > 0
            )
            
        except Exception as e:
            logger.error(f"Error in pool health assessment: {str(e)}")
            
        return health_metrics

    def _maintain_warm_pool(self, demand_forecast: DemandForecast) -> None:
        """Maintain warm container pool based on demand forecast"""
        try:
            target_warm_size = int(demand_forecast.predicted_demand * 
                                 self.config.get('warm_pool_factor', 0.2))
            
            current_warm_size = len(self.warm_pool)
            
            if current_warm_size < target_warm_size:
                # Warm up additional containers
                containers_to_warm = min(
                    target_warm_size - current_warm_size,
                    len(self.cold_pool)
                )
                
                for _ in range(containers_to_warm):
                    if self.cold_pool:
                        container_id = self.cold_pool.pop()
                        success = self._warm_up_container(container_id)
                        if success:
                            self.warm_pool.add(container_id)
                
            elif current_warm_size > target_warm_size:
                # Cool down excess containers
                containers_to_cool = current_warm_size - target_warm_size
                containers_to_remove = list(self.warm_pool)[:containers_to_cool]
                
                for container_id in containers_to_remove:
                    self.warm_pool.remove(container_id)
                    self.cold_pool.add(container_id)
                    self._cool_down_container(container_id)
            
        except Exception as e:
            logger.error(f"Error in warm pool maintenance: {str(e)}")

    def _cleanup_idle_containers(self) -> None:
        """Clean up idle containers exceeding idle timeout"""
        try:
            current_time = datetime.now()
            idle_timeout = timedelta(seconds=self.config.get('idle_timeout', 300))
            
            for container_id, last_used in self.container_last_used.items():
                if (current_time - last_used) > idle_timeout:
                    if container_id in self.warm_pool:
                        self.warm_pool.remove(container_id)
                        self.cold_pool.add(container_id)
                        self._cool_down_container(container_id)
                    
                    if self._can_remove_container(container_id):
                        self._cleanup_container(container_id)
            
        except Exception as e:
            logger.error(f"Error in idle container cleanup: {str(e)}")

    # Helper methods
    def _initialize_container(self, container_id: str) -> bool:
        """Initialize a new container"""
        try:
            container = self.containers[container_id]
            container.status = ContainerState.INITIALIZING
            
            # Simulate container initialization (replace with actual initialization logic)
            time.sleep(self.config.get('warmup_time', 1))
            
            container.status = ContainerState.RUNNING
            return True
            
        except Exception as e:
            logger.error(f"Error initializing container {container_id}: {str(e)}")
            return False

    def _warm_up_container(self, container_id: str) -> bool:
        """Warm up a container from cold pool"""
        try:
            container = self.containers[container_id]
            container.status = ContainerState.INITIALIZING
            
            # Simulate warm-up process (replace with actual warm-up logic)
            time.sleep(self.config.get('warmup_time', 1))
            
            container.status = ContainerState.RUNNING
            return True
            
        except Exception as e:
            logger.error(f"Error warming up container {container_id}: {str(e)}")
            return False

    def _cool_down_container(self, container_id: str) -> bool:
        """Cool down a container to cold pool"""
        try:
            container = self.containers[container_id]
            container.status = ContainerState.IDLE
            return True
            
        except Exception as e:
            logger.error(f"Error cooling down container {container_id}: {str(e)}")
            return False

    def _check_container_health(self, container: Container) -> Dict[str, Any]:
        """Check health status of a container"""
        health_status = {
            'healthy': True,
            'issues': []
        }
        
        try:
            metrics = self.container_metrics[container.id].collect_system_metrics()
            
            # Check resource usage
            if metrics.cpu_usage > self.config.get('max_cpu_usage', 0.9):
                health_status['issues'].append('high_cpu_usage')
                health_status['healthy'] = False
            
            if metrics.memory_usage > self.config.get('max_memory_usage', 0.9):
                health_status['issues'].append('high_memory_usage')
                health_status['healthy'] = False
            
            if metrics.error_rate > self.config.get('max_error_rate', 0.05):
                health_status['issues'].append('high_error_rate')
                health_status['healthy'] = False
            
        except Exception as e:
            logger.error(f"Error checking container health: {str(e)}")
            health_status['healthy'] = False
            health_status['issues'].append('health_check_failed')
        
        return health_status

    def _calculate_placement_cost(self, container: Container, node: Node,
                                remaining_resources: Dict[str, float]) -> float:
        """Calculate placement cost for a container on a node"""
        try:
            # Calculate resource utilization cost
            cpu_cost = (node.available_resources['cpu'] - remaining_resources['cpu']) / \
                      node.available_resources['cpu']
            memory_cost = (node.available_resources['memory'] - remaining_resources['memory']) / \
                         node.available_resources['memory']
            
            # Calculate load balancing cost
            container_count_cost = len(node.containers) / \
                                 self.config.get('max_containers_per_node', 10)
            
            # Combine costs with weights
            weights = {
                'cpu': 0.3,
                'memory': 0.3,
                'container_count': 0.4
            }
            
            total_cost = (
                weights['cpu'] * cpu_cost +
                weights['memory'] * memory_cost +
                weights['container_count'] * container_count_cost
            )
            
            return total_cost
            
        except Exception as e:
            logger.error(f"Error calculating placement cost: {str(e)}")
            return float('inf')

    def _sort_containers_by_priority(self, containers: List[Container]) -> List[Container]:
        """Sort containers by priority based on metrics and status"""
        try:
            container_scores = []
            
            for container in containers:
                score = self._calculate_container_priority(container)
                container_scores.append((container, score))
            
            sorted_containers = [c for c, _ in sorted(container_scores, 
                                                    key=lambda x: x[1], 
                                                    reverse=True)]
            return sorted_containers
            
        except Exception as e:
            logger.error(f"Error sorting containers by priority: {str(e)}")
            return containers

    def get_active_containers(self) -> List[Container]:
        """Get list of active containers"""
        return [container for container in self.containers.values()
                if container.status in {ContainerState.RUNNING, ContainerState.IDLE}]

    def _create_empty_metrics(self) -> Metrics:
        """Create empty metrics object"""
        return Metrics(
            cpu_usage=0.0,
            memory_usage=0.0,
            network_io=0.0,
            disk_io=0.0,
            request_count=0,
            response_time=0.0,
            error_rate=0.0
        )

    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current system-wide metrics"""
        try:
            active_containers = self.get_active_containers()
            if not active_containers:
                return {
                    'utilization': 0.0,
                    'error_rate': 0.0,
                    'response_time': 0.0
                }
            
            metrics = {
                'utilization': np.mean([c.metrics.cpu_usage for c in active_containers]),
                'error_rate': np.mean([c.metrics.error_rate for c in active_containers]),
                'response_time': np.mean([c.metrics.response_time for c in active_containers])
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting current metrics: {str(e)}")
            return {
                'utilization': 0.0,
                'error_rate': 0.0,
                'response_time': 0.0
            }