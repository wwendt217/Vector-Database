#ifndef DISTANCES_HPP
#define DISTANCES_HPP

#include <vector>

// Calculates the squared Euclidean distance between two vectors of floats.
static float defaultDistance(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    float squaredDistance = 0.0f;
    for (size_t i = 0; i < vec1.size(); ++i) {
        squaredDistance += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    }
    return squaredDistance; 
}

#endif // DISTANCES_HPP
