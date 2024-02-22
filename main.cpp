#include "HyperLogLog.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <unordered_set>
#include <cstdint> // Include for uint32_t

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <num_registers> <exponent for number of elements>" << endl;
        return 1;
    }

    uint32_t num_registers = static_cast<uint32_t>(std::atoi(argv[1]));
    uint32_t exponent = static_cast<uint32_t>(std::atoi(argv[2]));

    // Create a random device
    std::random_device rd;
    // Seed the generator
    std::mt19937 gen(rd());
    // Define the distribution range for 32-bit unsigned integers
    std::uniform_int_distribution<uint32_t> dis(0, std::numeric_limits<uint32_t>::max());
    std::uniform_int_distribution<uint32_t> repeat(0, 30);

    // Create an instance of HyperLogLog
    HyperLogLog hll(num_registers);

    // Example data to add and a set for testing
    std::vector<uint32_t> data;

    for (size_t i = 0; i < static_cast<uint32_t>(2 << exponent); i++) {
        uint32_t randomNumber = dis(gen);
        for (size_t j = 0; j < repeat(gen); j++) {
            data.push_back(randomNumber);
        }
    }

    // Push a known edge case to data for testing
    data.push_back(std::numeric_limits<uint32_t>::max());
    std::unordered_set<uint32_t> testSet(data.begin(), data.end());
    
    // Pass the data to HyperLogLog
    hll.add_bulk(data.data(), data.size());

    // Estimate the count of distinct elements
    float estimate = hll.estimateCountDistinct();
    int actual = testSet.size();

    // Output the estimated vs actual distinct count
    std::cout << std::fixed << "Estimated number of distinct elements: " << estimate << std::endl;
    std::cout << "Actual number of distinct elements: " << actual << std::endl;

    // Calculate and print the accuracy
    float error = abs(estimate - actual) / static_cast<float>(actual);
    std::cout << "Error: " << error << std::endl;

    return 0;
}