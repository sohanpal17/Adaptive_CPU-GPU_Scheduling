# Adaptive CPU-GPU Scheduling for Neural Network Training

An adaptive layer-wise scheduling system that distributes neural network computation across CPU and GPU based on computational benefit, memory availability, execution time, energy estimates, and data-transfer overhead.

## Overview

Training deep learning models on GPUs can be challenging when available VRAM is limited. CPU-only training avoids GPU memory constraints but can be significantly slower. Meanwhile, assigning every layer to a GPU is not always the most efficient strategy.

This project explores **adaptive CPU-GPU scheduling**, where individual neural network units are assigned to either the CPU or GPU according to their measured performance and resource requirements.

Instead of relying on parameter count or assigning the entire model to one device, the scheduler profiles each unit, evaluates possible placements, and selects a configuration that balances performance and resource consumption.

The system also supports dynamic reassignment when the available GPU memory budget changes during training.

## Key Features

- **Layer-wise profiling:** Measures CPU and GPU execution time, memory footprint, and computational requirements for individual model units.
- **Adaptive scheduling:** Automatically determines whether each unit should execute on the CPU or GPU.
- **Compute-aware placement:** Uses measured GPU speedup rather than parameter count alone to identify GPU candidates.
- **Memory-aware scheduling:** Selects placements that satisfy available GPU memory constraints.
- **Transfer overhead estimation:** Measures CPU-GPU data-transfer latency and bandwidth to account for communication costs.
- **Hybrid execution:** Executes different parts of the same neural network across CPU and GPU while preserving gradient flow.
- **Dynamic memory adaptation:** Monitors resource availability and can migrate units between devices during training.
- **Optimizer-state migration:** Moves model parameters and optimizer states together to maintain training consistency.
- **Performance evaluation:** Compares execution time, GPU memory usage, energy, throughput, and accuracy.
- **Explainable decisions:** Records placement decisions and the reasons behind migrations.

## How It Works

The system follows a profiling-and-scheduling pipeline:

1. **Model Analysis:** Divides the CNN into 11 independently schedulable units and analyzes parameters, FLOPs, activation sizes, and memory requirements.
2. **Hardware Detection:** Detects CPU and GPU resources, available VRAM, GPU utilization, and power information.
3. **Performance Profiling:** Executes each unit on both CPU and GPU to measure execution time, memory consumption, and power-related metrics.
4. **Transfer Modeling:** Measures host-to-device and device-to-host transfer costs.
5. **Cost Estimation:** Estimates the time, energy, GPU cost, and memory requirements of candidate placements.
6. **Placement Optimization:** Evaluates possible CPU-GPU assignments and selects a feasible configuration.
7. **Hybrid Execution:** Executes the network according to the selected placement while maintaining autograd and optimizer-state consistency.
8. **Dynamic Re-evaluation:** Periodically updates performance estimates and migrates units when the expected benefit justifies the migration cost.

## Scheduling Strategy

The scheduler evaluates candidate placements using a weighted objective function:

`J = αT + βE + γC + δM`

Where:

- **T:** Normalized training execution time.
- **E:** Normalized energy consumption.
- **C:** Normalized GPU usage cost.
- **M:** Normalized GPU memory consumption.

Initial objective weights:

| Metric | Weight |
|---|---:|
| Training time (α) | 0.30 |
| Energy (β) | 0.25 |
| GPU cost (γ) | 0.10 |
| GPU memory (δ) | 0.35 |

The scheduler also applies two important rules:

- **Hard constraint:** GPU memory usage must remain within the available memory budget.
- **Soft constraint:** Predicted execution time should remain within an acceptable slowdown relative to the all-GPU configuration.

A configurable minimum GPU speedup threshold of **1.25×** is initially used to determine whether a unit is eligible for GPU placement.

The scheduler evaluates complete placements rather than making decisions using parameter count alone. This is important because a layer with many parameters does not necessarily require the GPU, while a smaller convolutional layer may benefit considerably from GPU parallelism.

## Dynamic Adaptation

The system periodically monitors execution and reassesses the current placement.

For example:

- Training begins with a particular GPU memory budget.
- Another process consumes GPU memory, reducing the available capacity.
- The monitoring system detects the change.
- The scheduler recalculates the placement.
- Selected units are migrated from GPU to CPU when necessary.
- Training continues without restarting.

Migration takes place at safe iteration boundaries. Model weights and optimizer states are moved together to prevent device inconsistencies.

## Experimental Setup

The project uses a CNN trained on the CIFAR dataset.

**Model architecture:**
- 8 convolutional units.
- 3 fully connected units.
- 11 independently schedulable units in total.

**Dataset:**
- CIFAR-10 or CIFAR-100, configurable in the notebook.
- CIFAR-10 contains 50,000 training images and 10,000 test images.
- Images are normalized and converted into the format expected by PyTorch.

**Training configuration:**
- Batch size: 128.
- Optimizer: Adam.
- Learning rate: 0.0005.
- Configurable model width multiplier.
- Identical initial weights and deterministic batch ordering for comparable experiments.

## GPU Memory Experiments

The system evaluates five GPU memory budgets using the same physical GPU.

| Experiment | GPU Memory Budget |
|---|---:|
| E1 | 500 MB |
| E2 | 1 GB |
| E3 | 2 GB |
| E4 | 4 GB |
| E5 | Maximum available GPU memory |

Three scheduling strategies are used for comparison:

1. **GPU-only:** All model units execute on the GPU. Configurations that exceed the enforced PyTorch allocator limit are recorded as OOM.
2. **CPU-only:** All model units execute on the CPU. This provides a CPU performance baseline.
3. **Adaptive CPU-GPU:** The scheduler distributes units between CPU and GPU according to profiling results, memory constraints, and the objective function.

Even in E5, where more GPU memory is available, the adaptive scheduler continues evaluating hybrid placements rather than automatically assigning every unit to the GPU.

## Evaluation Metrics

The project evaluates the following metrics:

| Metric | Purpose |
|---|---|
| Training time | Measures total execution time |
| Peak GPU memory | Measures maximum PyTorch allocator memory |
| GPU utilization | Evaluates GPU activity |
| CPU utilization | Evaluates CPU activity |
| Energy consumption | Measures or estimates energy usage |
| Throughput | Measures training iterations per second |
| Transfer overhead | Estimates CPU-GPU communication cost |
| Scheduling overhead | Measures reassessment and migration costs |
| Model accuracy | Evaluates prediction performance |
| Placement decisions | Shows which units execute on each device |

Measurements are distinguished from estimates. GPU power and energy are measured through NVIDIA monitoring when available, while CPU power is estimated using configurable assumptions. The GPU allocator peak does not include the CUDA context.

## Technology Stack

- **Programming Language:** Python
- **Deep Learning Framework:** PyTorch
- **GPU Computing:** CUDA
- **Dataset:** CIFAR-10 / CIFAR-100
- **Numerical Computing:** NumPy
- **Data Analysis:** Pandas
- **Visualization:** Matplotlib
- **GPU Monitoring:** NVIDIA Management Library (NVML)
- **System Monitoring:** psutil
- **Development Environment:** Jupyter Notebook

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch torchvision numpy pandas matplotlib psutil pynvml jupyter
```

Install a PyTorch build compatible with your NVIDIA GPU and CUDA driver. Refer to the official PyTorch installation guide:

https://pytorch.org/get-started/locally/

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

`ML-project_Adaptive_CPU_GPU_Scheduling.ipynb`

### 5. Run the notebook

- Ensure that PyTorch can detect your CUDA GPU.
- Run the notebook cells sequentially.
- Allow the dataset to download when prompted by the notebook.
- Wait for CPU and GPU profiling to finish.
- Execute the E1-E5 experiments.
- Inspect the generated tables, graphs, placement decisions, and final evaluation.

**Important:** A CUDA-enabled NVIDIA GPU is required for the current notebook implementation. Google Colab with a GPU runtime can also be used.

## Project Structure

```text
Adaptive-CPU-GPU-Scheduling/
│
├── ML-project_Adaptive_CPU_GPU_Scheduling.ipynb
│
├── results_summary.csv
│
└── README.md
```

`results_summary.csv` is generated by the notebook after the experiments complete.

## Important Experimental Considerations

- GPU memory budgets are enforced through PyTorch's per-process CUDA allocator limit. They are not equivalent to physically reducing GPU VRAM, and CUDA context memory is outside that allocator limit.
- CPU energy values are estimates, not direct hardware measurements.
- The GPU cost term uses placeholder pricing and should be configured for the target environment.
- CPU-only results may be extrapolated from shorter runs unless the full CPU experiment is enabled. Extrapolated results must not be treated as fully measured results.
- The experiments use a fixed number of training iterations for controlled comparisons. They are not intended to demonstrate fully converged model accuracy.
- Hybrid execution introduces communication and scheduling overhead. It is not guaranteed to outperform GPU-only execution in training time or energy consumption.
- The current exhaustive placement search supports up to 16 schedulable units. Larger models will require more scalable optimization methods.

## Project Objective

The primary objective is to investigate whether intelligent layer-wise CPU-GPU scheduling can improve GPU memory efficiency while maintaining computational correctness and acceptable training performance.

Rather than assuming that every neural network operation belongs on the GPU, this project treats hardware placement as an optimization problem involving computation, memory, communication, and energy trade-offs.

## Future Scope

- Extend scheduling to larger CNNs and transformer architectures.
- Explore graph-aware scheduling for models with skip connections and branching.
- Investigate more scalable placement optimization algorithms.
- Improve CPU energy measurement using hardware-level energy counters.
- Support multi-GPU scheduling.
- Explore predictive scheduling using workload and memory-usage trends.
- Evaluate scheduling across different GPU architectures and hardware configurations.

## Author

**Adaptive CPU-GPU Scheduling Research Project**

Developed as an experimental research prototype exploring intelligent heterogeneous computing for deep learning workloads.