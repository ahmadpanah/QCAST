# QCAST: Q-Learning Container Adaptive Scheduling Technique

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)

QCAST is an intelligent container scheduling and resource optimization system for serverless platforms, leveraging Q-Learning for adaptive decision-making. It provides automated scaling, resource optimization, and workload analysis capabilities to optimize serverless application performance.

## 📋 Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Components](#-components)
- [Examples](#-examples)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- **Intelligent Container Scheduling**: Q-Learning based decision-making for optimal container placement and scaling
- **Dynamic Resource Optimization**: Automated resource allocation based on workload patterns and performance metrics
- **Workload Analysis**: Advanced pattern detection and anomaly identification
- **Adaptive Control**: Self-adjusting parameters based on system performance
- **Performance Monitoring**: Comprehensive metrics collection and analysis
- **Fault Tolerance**: Built-in error handling and system recovery mechanisms

## 🏗 System Architecture

QCAST consists of four main components:

1. **Q-Learning Engine**: Implements reinforcement learning for decision-making
2. **Container Lifecycle Manager**: Handles container operations and scheduling
3. **Resource Optimizer**: Manages resource allocation and optimization
4. **Workload Monitor**: Analyzes workload patterns and system metrics

<!-- ![QCAST Architecture](docs/images/architecture.png) -->

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip
- Virtual environment (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/ahmadpanah/qcast.git
cd qcast

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Quick Start

```python
from qcast.core.qcast_system import QCASTSystem

# Initialize QCAST system
qcast = QCASTSystem("config/qcast_config.json")

# Start the system
qcast.run()

# Monitor system status
status = qcast.get_system_status()
print(status)

# Cleanup
qcast.cleanup()
```

## ⚙️ Configuration

QCAST uses a JSON configuration file for system settings. Example configuration:

```json
{
    "q_learning": {
        "learning_rate": 0.1,
        "discount_factor": 0.95,
        "initial_epsilon": 1.0
    },
    "container": {
        "min_containers": 1,
        "max_containers": 100
    },
    "resources": {
        "optimization_strategy": "balanced",
        "update_interval": 60
    },
    "monitoring": {
        "interval": 10,
        "window_size": 24
    }
}
```

## 🧩 Components

### Q-Learning Engine

```python
from qcast.components.q_learning_engine import QLearningEngine

engine = QLearningEngine(config)
action = engine.select_action(state, valid_actions)
```

### Container Lifecycle Manager

```python
from qcast.components.container_lifecycle_manager import ContainerLifecycleManager

manager = ContainerLifecycleManager(config)
manager.manage_container_pool(current_demand, prediction_window, resource_constraints)
```

### Resource Optimizer

```python
from qcast.components.resource_optimizer import ResourceOptimizer

optimizer = ResourceOptimizer(config)
optimized_resources = optimizer.optimize_resources(container, metrics, thresholds)
```

### Workload Monitor

```python
from qcast.components.workload_monitor import WorkloadMonitor

monitor = WorkloadMonitor(config)
analysis = monitor.analyze_workload(history, window_size)
```

## 📝 Examples

### Basic Usage

```python
# Example configuration
config = {
    "q_learning": {
        "learning_rate": 0.1,
        "discount_factor": 0.95
    }
}

# Initialize system
qcast = QCASTSystem("config/qcast_config.json")

# Run system
qcast.run()
```

See [examples](examples/) directory for more examples.

## 📊 Project Structure

```
qcast/
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── data_models.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── q_learning_engine.py
│   │   ├── container_lifecycle_manager.py
│   │   ├── resource_optimizer.py
│   │   └── workload_monitor.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── qcast_system.py
│   └── utils/
│       ├── __init__.py
│       └── metrics.py
││
├── examples/
│   ├── usage_example.py
│
│
├── config/
│   └── qcast_config.json
│
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```


## 📈 Performance Metrics

QCAST provides comprehensive performance monitoring:

- Container scaling efficiency
- Resource utilization
- Response time optimization
- Error rate monitoring
- System overhead

## Dependencies

The following packages are required:

```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
statsmodels>=0.13.0
scikit-learn>=0.24.0
psutil>=5.8.0
pytest>=6.2.5
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📧 Contact

Seyed Hossein Ahmadpanah - h.ahmadpanah@iau.ac.ir

Project Link: [https://github.com/ahmadpanah/qcast](https://github.com/yourusername/qcast)

## 🔗 Related Projects

- [Serverless Framework](https://github.com/serverless/serverless)
- [OpenFaaS](https://github.com/openfaas/faas)
- [Knative](https://github.com/knative/serving)

---
Made with ❤️ by [Seyed Hossein Ahmadpanah](https://github.com/ahmadpanah)