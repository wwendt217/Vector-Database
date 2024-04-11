#ifndef NORMALIZATIONQUANTIZER_HPP
#define NORMALIZATIONQUANTIZER_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

class NormalizationQuantizer {
private:
    float mean = 0.0f;
    float stdDev = 1.0f; // Default to 1 to prevent division by zero
    int numBins;
    float binSize;
    float minValue;
    float maxValue;

public:
    NormalizationQuantizer(int numBins)
        : numBins(numBins) {
        if (numBins <= 0) {
            throw std::invalid_argument("numBins must be greater than 0");
        }
    }

    // Learns the normalization parameters (mean and standard deviation) from the data
    void learnNormalizationParameters(const std::vector<float>& data) {
        if (data.empty()) {
            throw std::invalid_argument("Data for learning normalization parameters cannot be empty");
        }
        
        mean = std::accumulate(data.begin(), data.end(), 0.0f) / data.size();
        
        float variance = std::accumulate(data.begin(), data.end(), 0.0f, [this](float acc, float val) {
            return acc + (val - mean) * (val - mean);
        }) / data.size();
        
        stdDev = std::sqrt(variance);

        // Assuming normalized data is within 3 standard deviations from the mean
        minValue = -3.0f;
        maxValue = 3.0f;
        binSize = (maxValue - minValue) / numBins;
    }

    // Quantizes a normalized value into a bin index
    int quantize(float value) const {
        // Normalize the value
        float normalizedValue = (value - mean) / stdDev;
        // Quantize the normalized value
        int bin = static_cast<int>((normalizedValue - minValue) / binSize);
        return std::min(std::max(bin, 0), numBins - 1); // Ensure bin is within range
    }
};

#endif // NORMALIZATIONQUANTIZER_HPP
