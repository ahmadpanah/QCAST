# QCAST: Q-Learning Container Adaptive Scheduling Technique

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](docs/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

QCAST is an intelligent container scheduling and resource optimization system for serverless platforms, leveraging Q-Learning for adaptive decision-making. It provides automated scaling, resource optimization, and workload analysis capabilities to optimize serverless application performance.

## 📋 Table of Contents

- [QCAST: Q-Learning Container Adaptive Scheduling Technique](#qcast-q-learning-container-adaptive-scheduling-technique)
  - [📋 Table of Contents](#-table-of-contents)
  - [✨ Features](#-features)
  - [🏗 System Architecture](#-system-architecture)
  - [🚀 Installation](#-installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
- [Clone the repository](#clone-the-repository)
- [Create and activate virtual environment](#create-and-activate-virtual-environment)
- [or](#or)
- [Install dependencies](#install-dependencies)

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

![QCAST Architecture](docs/images/architecture.png)

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip
- Virtual environment (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/ahmadpanah/QCAST
cd qcast

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt