#ifndef HYPERLOGLOG_CUH_
#define HYPERLOGLOG_CUH_

#include <cuda_runtime.h>
#include <cstdint>

__device__ uint32_t getLeftMostBitsValue(uint32_t n, uint32_t k);
__device__ uint32_t getRank(uint32_t w);
__global__ void kernelCalculateRanks(uint32_t* data, uint32_t* ranks, uint32_t size, uint32_t num_bucket_bits);
__global__ void processRanksAndUpdateRegisters(uint32_t* data, uint32_t* ranks, uint32_t* registers, uint32_t dataSize, uint32_t num_registers, uint32_t num_bucket_bits);

class HyperLogLog {
public:
    HyperLogLog(uint32_t num_registers);
    ~HyperLogLog();
    void add_bulk(uint32_t* arr, uint32_t size);
    float estimateCountDistinct();

private:
    uint32_t* h_registers;
    uint32_t num_registers;
    float alpha_m;
    uint32_t num_bucket_bits;

    uint32_t* allocateGPUMemory(uint32_t size);
    void freeGPUMemory(uint32_t* arr);
    void memCopy(uint32_t* hostData, uint32_t* deviceData, uint32_t size, cudaMemcpyKind kind);
    float getAlpha(uint32_t num_registers);
};

#endif // HYPERLOGLOG_CUH_