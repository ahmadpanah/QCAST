# examples/usage_example.py
from datetime import datetime, timedelta
from src.core.qcast_system import QCAST
from src.models.data_models import Metrics

def main():
    config = {
        'min_containers': 1,
        'max_containers': 100,
        'container_capacity': 100,
        'safety_factor': 1.2,
        'cpu_per_container': 0.1,
        'memory_per_container': 256,
        'network_per_container': 100,
        'scale_up_threshold': 0.8,
        'scale_down_threshold': 0.3,
        'thresholds': {
            'max_error_rate': 0.01,
            'target_error_rate': 0.001,
            'max_response_time': 1000,
            'target_response_time': 100
        }
    }

    qcast = QCAST(config)
    qcast.run()

if __name__ == "__main__":
    main()