# src/utils/metrics.py
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import numpy as np
from dataclasses import asdict
from collections import deque
import psutil
import time

from ..models.data_models import Metrics

class MetricsCollector:
    def __init__(self, history_size: int = 1000):

        self.history_size = history_size
        self.history = deque(maxlen=history_size)
        self.last_collection_time = None

    def collect_system_metrics(self) -> Metrics:

        current_time = time.time()
        
        # Collect CPU metrics
        cpu_usage = psutil.cpu_percent(interval=None) / 100.0

        # Collect memory metrics
        memory = psutil.virtual_memory()
        memory_usage = memory.percent / 100.0

        # Collect network I/O metrics
        network = psutil.net_io_counters()
        network_io = (network.bytes_sent + network.bytes_recv) / 1024 / 1024  # MB

        # Collect disk I/O metrics
        disk = psutil.disk_io_counters()
        disk_io = (disk.read_bytes + disk.write_bytes) / 1024 / 1024  # MB

        # Calculate rates if we have previous measurements
        if self.last_collection_time is not None:
            time_diff = current_time - self.last_collection_time
            network_io /= time_diff
            disk_io /= time_diff

        self.last_collection_time = current_time

        metrics = Metrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            network_io=network_io,
            disk_io=disk_io,
            request_count=0,  # To be updated by the application
            response_time=0,  # To be updated by the application
            error_rate=0      # To be updated by the application
        )

        self.history.append((datetime.now(), metrics))
        return metrics

class MetricsAnalyzer:
    """Analyzes metrics for patterns, anomalies, and trends"""
    
    def __init__(self, window_size: int = 60):

        self.window_size = window_size
        self.anomaly_threshold = 2.0  # Standard deviations for anomaly detection

    def calculate_statistics(self, metrics_history: List[Tuple[datetime, Metrics]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate basic statistics for each metric over the history.
        
        Args:
            metrics_history: List of timestamped metrics
        
        Returns:
            Dict containing statistics for each metric
        """
        if not metrics_history:
            return {}

        metrics_dict = {}
        metric_fields = asdict(metrics_history[0][1]).keys()

        for field in metric_fields:
            values = [getattr(m, field) for _, m in metrics_history]
            metrics_dict[field] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99)
            }

        return metrics_dict

    def detect_anomalies(self, metrics_history: List[Tuple[datetime, Metrics]]) -> Dict[str, List[datetime]]:
        """
        Detect anomalies in metrics using statistical analysis.
        
        Args:
            metrics_history: List of timestamped metrics
        
        Returns:
            Dict mapping metric names to lists of anomaly timestamps
        """
        anomalies = {}
        if len(metrics_history) < self.window_size:
            return anomalies

        stats = self.calculate_statistics(metrics_history[-self.window_size:])
        
        for field in asdict(metrics_history[0][1]).keys():
            anomalies[field] = []
            mean = stats[field]['mean']
            std = stats[field]['std']
            
            for timestamp, metrics in metrics_history[-self.window_size:]:
                value = getattr(metrics, field)
                if abs(value - mean) > self.anomaly_threshold * std:
                    anomalies[field].append(timestamp)

        return anomalies

    def calculate_trends(self, metrics_history: List[Tuple[datetime, Metrics]], 
                        window_minutes: int = 60) -> Dict[str, float]:
        """
        Calculate trends for each metric over specified time window.
        
        Args:
            metrics_history: List of timestamped metrics
            window_minutes: Time window for trend calculation in minutes
        
        Returns:
            Dict mapping metric names to trend values
        """
        if not metrics_history:
            return {}

        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = [(t, m) for t, m in metrics_history if t >= cutoff_time]
        
        if len(recent_metrics) < 2:
            return {}

        trends = {}
        metric_fields = asdict(recent_metrics[0][1]).keys()

        for field in metric_fields:
            times = [(t - recent_metrics[0][0]).total_seconds() for t, _ in recent_metrics]
            values = [getattr(m, field) for _, m in recent_metrics]
            
            if len(set(values)) == 1:  # All values are the same
                trends[field] = 0.0
            else:
                # Calculate linear regression slope
                coefficients = np.polyfit(times, values, 1)
                trends[field] = coefficients[0]  # Slope represents trend

        return trends

class PerformanceCalculator:
    """Calculates various performance metrics and SLA compliance"""
    
    def __init__(self, sla_thresholds: Dict[str, float]):

        self.sla_thresholds = sla_thresholds

    def calculate_sla_compliance(self, metrics_history: List[Tuple[datetime, Metrics]]) -> Dict[str, float]:

        if not metrics_history:
            return {}

        compliance = {}
        total_points = len(metrics_history)

        for metric_name, threshold in self.sla_thresholds.items():
            if hasattr(metrics_history[0][1], metric_name):
                compliant_points = sum(
                    1 for _, m in metrics_history
                    if getattr(m, metric_name) <= threshold
                )
                compliance[metric_name] = compliant_points / total_points

        return compliance

    def calculate_performance_score(self, metrics: Metrics) -> float:

        scores = []
        
        # CPU score
        cpu_score = 1.0 - metrics.cpu_usage
        scores.append(cpu_score)
        
        # Memory score
        memory_score = 1.0 - metrics.memory_usage
        scores.append(memory_score)
        
        # Response time score (normalized to 0-1)
        max_acceptable_response_time = self.sla_thresholds.get('response_time', 1000)
        response_time_score = max(0, 1 - (metrics.response_time / max_acceptable_response_time))
        scores.append(response_time_score)
        
        # Error rate score
        max_acceptable_error_rate = self.sla_thresholds.get('error_rate', 0.01)
        error_rate_score = max(0, 1 - (metrics.error_rate / max_acceptable_error_rate))
        scores.append(error_rate_score)
        
        # Calculate weighted average (can be adjusted based on importance)
        weights = [0.3, 0.2, 0.3, 0.2]  # CPU, Memory, Response Time, Error Rate
        return sum(score * weight for score, weight in zip(scores, weights))

class MetricsFormatter:
    """Formats metrics for logging and reporting"""
    
    @staticmethod
    def format_metrics_for_logging(metrics: Metrics) -> str:

        return (
            f"CPU: {metrics.cpu_usage*100:.1f}% | "
            f"Memory: {metrics.memory_usage*100:.1f}% | "
            f"Network I/O: {metrics.network_io:.1f}MB/s | "
            f"Disk I/O: {metrics.disk_io:.1f}MB/s | "
            f"Requests: {metrics.request_count} | "
            f"Response Time: {metrics.response_time:.1f}ms | "
            f"Error Rate: {metrics.error_rate*100:.2f}%"
        )

    @staticmethod
    def format_metrics_for_prometheus(metrics: Metrics, labels: Dict[str, str] = None) -> List[str]:

        label_str = ""
        if labels:
            label_str = "{" + ",".join(f'{k}="{v}"' for k, v in labels.items()) + "}"

        return [
            f"system_cpu_usage{label_str} {metrics.cpu_usage}",
            f"system_memory_usage{label_str} {metrics.memory_usage}",
            f"system_network_io_bytes{label_str} {metrics.network_io}",
            f"system_disk_io_bytes{label_str} {metrics.disk_io}",
            f"system_request_count{label_str} {metrics.request_count}",
            f"system_response_time_milliseconds{label_str} {metrics.response_time}",
            f"system_error_rate{label_str} {metrics.error_rate}"
        ]

# Example usage
def example_usage():
    # Initialize components
    collector = MetricsCollector()
    analyzer = MetricsAnalyzer()
    calculator = PerformanceCalculator({
        'response_time': 1000,  # ms
        'error_rate': 0.01,     # 1%
        'cpu_usage': 0.8,       # 80%
        'memory_usage': 0.8     # 80%
    })
    formatter = MetricsFormatter()

    # Collect metrics
    metrics = collector.collect_system_metrics()

    # Analyze metrics
    stats = analyzer.calculate_statistics(collector.history)
    anomalies = analyzer.detect_anomalies(collector.history)
    trends = analyzer.calculate_trends(collector.history)

    # Calculate performance
    sla_compliance = calculator.calculate_sla_compliance(collector.history)
    performance_score = calculator.calculate_performance_score(metrics)

    # Format metrics
    log_output = formatter.format_metrics_for_logging(metrics)
    prometheus_output = formatter.format_metrics_for_prometheus(metrics, {'service': 'qcast'})

    # Print results
    print("Current Metrics:")
    print(log_output)
    print("\nPerformance Score:", performance_score)
    print("\nPrometheus Format:")
    for line in prometheus_output:
        print(line)

if __name__ == "__main__":
    example_usage()