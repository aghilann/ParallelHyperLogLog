# HyperLogLog CUDA Implementation

## Overview
This project implements the HyperLogLog algorithm for distinct count estimation utilizing CUDA for parallel processing. The focus is on optimizing performance through CUDA kernels while maintaining the same accurancy as the sequential implementation in count estimation. The real challenge is in maximizing the throughput of the parallel implementation, which is the primary goal of this project.

## Features
- **CUDA Acceleration:** Utilizes CUDA kernels for efficient parallel computation of large datasets. Ideal for batch processing and offline evaluation.
- **High Accuracy:** Achieves an average accuracy of 97%, closely matching the sequential implementation benchmark set by Google's HyperLogLog.
- **Dynamic Input:** Very large throughput for processing large datasets, with the ability to process millions of records in seconds.

## Getting Started
### Prerequisites
- NVIDIA CUDA Toolkit
- A CUDA-capable NVIDIA GPU
- C++ compiler with CUDA support

### Compilation
Use the provided Makefile for compilation:
```shell
make
```

Usage

```shell
./HyperLogLog <num_registers> <bit_shift_exponent> # execute the binary directly

cd python
python3 main.py # or python main.py
```

### Implementation Details
The add_bulk method is a key component, designed to efficiently process bulk data inputs. By leveraging CUDA's parallel computing capabilities, it distributes data processing across multiple threads, significantly speeding up the estimation process.

### Accuracy

The implementation has a 2.43% standard error measurement to quantify the precision of estimations. This performance is within 2% of Google's sequential HyperLogLog implementation (2% standard error).

### Throughout Benchmarks

I am currently working on a benchmarking suite to evaluate the performance of the CUDA implementation's throughput which is the most exciting part. The current code will need to be modified since the current binary also performs the sequential execution.

