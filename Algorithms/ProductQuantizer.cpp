#include "ProductQuantizer.hpp"
#include <iostream>
#include <cstdlib>
#include <ctime>

// Function to generate a random vector of floats
std::vector<float> generateRandomVector(size_t length) {
    std::vector<float> vec(length);
    std::generate(vec.begin(), vec.end(), []() -> float {
        return static_cast<float>(rand()) / RAND_MAX; // Generate a float in [0, 1)
    });
    return vec;
}

// Define projection functions
std::vector<float> projectFirstHalf(const std::vector<float>& vec) {
    return std::vector<float>(vec.begin(), vec.begin() + vec.size() / 2);
}

std::vector<float> projectSecondHalf(const std::vector<float>& vec) {
    return std::vector<float>(vec.begin() + vec.size() / 2, vec.end());
}

int main() {
    srand(static_cast<unsigned int>(time(nullptr)));

    constexpr int vectorLength = 8; // Total length of each vector
    constexpr int numCentroids = 4; // Number of centroids for KNN in each subspace
    constexpr int numSubspaces = 2; // Dividing the vector into 2 subspaces
    constexpr int numDataPoints = 100; // Number of data points to generate for training
    constexpr int numTestPoints = 5; // Number of test points to generate

    // Define projection functions for each subspace
    std::vector<ProductQuantizer<vectorLength, numCentroids, numSubspaces>::ProjectionFunction> projections = {
        projectFirstHalf, projectSecondHalf
    };

    // Initialize ProductQuantizer with projection functions
    ProductQuantizer<vectorLength, numCentroids, numSubspaces> pq(projections);

    // Generate random data points for training
    std::vector<std::vector<float>> trainingData;
    for (int i = 0; i < numDataPoints; ++i) {
        trainingData.push_back(generateRandomVector(vectorLength));
    }

    // Train the ProductQuantizer with the generated data
    pq.train(trainingData);

    // Generate random test points and quantize them
    std::cout << "Quantizing test vectors:\n";
    for (int i = 0; i < numTestPoints; ++i) {
        auto testVector = generateRandomVector(vectorLength);
        auto quantizedIndices = pq.quantize(testVector);

        std::cout << "Test Vector " << i + 1 << ": [";
        for (const auto& elem : testVector) std::cout << elem << " ";
        std::cout << "] - Quantized Indices: [";
        for (const auto& index : quantizedIndices) std::cout << index << " ";
        std::cout << "]\n";
    }

    return 0;
}
