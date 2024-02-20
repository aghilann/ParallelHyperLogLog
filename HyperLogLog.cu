#include "HyperLogLog.cuh"
#include <iostream>
#include "CudaException.h"
#include <cmath>
#include <cstdint>

__global__ void processRanksAndUpdateRegisters(uint32_t* data, uint32_t* ranks, uint32_t* registers, uint32_t dataSize, uint32_t num_registers, uint32_t num_bucket_bits) {
    extern __shared__ uint32_t shared[];
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= dataSize) return;
    uint32_t threadRank = ranks[idx];
    uint32_t sharedIdx = getLeftMostBitsValue(data[idx], num_bucket_bits);
    if(threadIdx.x < num_registers) {
        shared[threadIdx.x] = 0; // Changed from INT_MIN to 0 for unsigned
    }
    __syncthreads();
    atomicMax(&shared[sharedIdx], threadRank);
    __syncthreads();
    if(threadIdx.x < num_registers) {
        atomicMax(&registers[sharedIdx], shared[sharedIdx]);
    }
}

__device__ uint32_t getLeftMostBitsValue(uint32_t n, uint32_t k) {
    const uint32_t totalBits = 8 * sizeof(uint32_t);
    uint32_t value = (n >> (totalBits - k)) & ((1 << k) - 1);
    return value;
}

__device__ uint32_t getRank(uint32_t w) {
    if (w == 0) return 32;
    uint32_t rank = 0;
    while ((w & 1) == 0 && rank < 32) {
        rank++;
        w >>= 1;
    }
    return rank;
}

__global__ void kernelCalculateRanks(uint32_t* data, uint32_t* ranks, uint32_t size, uint32_t num_bucket_bits) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;
    uint32_t rank = getRank(data[idx]);
    ranks[idx] = rank;
}

// Constructor
HyperLogLog::HyperLogLog(uint32_t num_registers) {
    this->num_registers = num_registers;
    this->h_registers = new uint32_t[this->num_registers]();
    this->num_bucket_bits = static_cast<uint32_t>(std::log2(num_registers));
}

// Destructor
HyperLogLog::~HyperLogLog() {
    delete[] this->h_registers;
}

// Allocates GPU memory
uint32_t* HyperLogLog::allocateGPUMemory(uint32_t size) {
    uint32_t* deviceArray = nullptr;
    cudaError_t cudaStatus = cudaMalloc((void**)&deviceArray, size * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        throw CudaException("CUDA malloc failed: " + std::string(cudaGetErrorString(cudaStatus)));
    }
    return deviceArray;
}

// Copies memory between host and device
void HyperLogLog::memCopy(uint32_t * hostData, uint32_t * deviceData, uint32_t size, cudaMemcpyKind kind) {
    cudaError_t cudaStatus;
    if (kind == cudaMemcpyHostToDevice) {
     cudaStatus = cudaMemcpy(deviceData, hostData, size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    } else if (kind == cudaMemcpyDeviceToHost) {
        cudaStatus = cudaMemcpy(hostData, deviceData, size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    } else {
        throw CudaException("Unsupported memcpy: " + std::string(cudaGetErrorString(cudaStatus)));
    }
}

// Frees GPU memory
void HyperLogLog::freeGPUMemory(uint32_t* arr) {
    cudaError_t cudaStatus = cudaFree(arr);
    if (cudaStatus != cudaSuccess) {
        throw CudaException("CUDA cudaFree failed: " + std::string(cudaGetErrorString(cudaStatus)));
    }
}

// Adds bulk data to the estimator
void HyperLogLog::add_bulk(uint32_t* hostData, uint32_t size) {
    uint32_t* kernelData = allocateGPUMemory(size);
    uint32_t* kernelRank = allocateGPUMemory(size); // Adjust type to uint32_t
    uint32_t* kernelRegisters = allocateGPUMemory(this->num_registers); // Adjust type to uint32_t

    cudaMemset(kernelRegisters, 0, this->num_registers * sizeof(uint32_t));
    memCopy(hostData, kernelData, size, cudaMemcpyHostToDevice);

    uint32_t threadsPerBlock = 1024;
    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    kernelCalculateRanks<<<blocks, threadsPerBlock>>>(kernelData, kernelRank, size, this->num_bucket_bits);
    cudaDeviceSynchronize();

    processRanksAndUpdateRegisters<<<blocks, threadsPerBlock, this->num_registers * sizeof(uint32_t)>>>(
        kernelData,
        kernelRank, 
        kernelRegisters,
        size, 
        this->num_registers, 
        this->num_bucket_bits);
    cudaDeviceSynchronize();

    memCopy(this->h_registers, kernelRegisters, this->num_registers, cudaMemcpyDeviceToHost);

    for (uint32_t i = 0; i < this->num_registers; i++) {
        std::cout << "Bucket " << i << " has value: " << this->h_registers[i] << std::endl;
    }

    freeGPUMemory(kernelData);
    freeGPUMemory(kernelRank);
    freeGPUMemory(kernelRegisters);
}

// Estimates the count of distinct elements
float HyperLogLog::estimateCountDistinct() {
    double alpha_m = getAlpha(this->num_registers);
    double Z_inverse = 0.0;
    for (uint32_t i = 0; i < this->num_registers; ++i) {
        Z_inverse += std::pow(2.0, -static_cast<double>(this->h_registers[i]));
    }
    double Z = 2.0 / Z_inverse;
    double raw_estimate = alpha_m * this->num_registers * this->num_registers * Z;

    if (raw_estimate <= (5.0 / 2.0) * this->num_registers) {
        uint32_t V = 0; // Count zeros in registers
        for (uint32_t i = 0; i < this->num_registers; ++i) {
            if (this->h_registers[i] == 0) V++;
        }
        if (V > 0) raw_estimate = this->num_registers * std::log(static_cast<double>(this->num_registers) / V);
    }
    return static_cast<float>(raw_estimate);
}

// Calculates alpha_m constant based on the number of registers
float HyperLogLog::getAlpha(uint32_t num_registers) {
    switch (num_registers) {
        case 16: return 0.673f;
        case 32: return 0.697f;
        case 64: return 0.709f;
        default: return 0.7213f / (1.0f + 1.079f / num_registers);
    }
}
