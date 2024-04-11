#ifndef VECTORSEARCHALGORITHM_HPP
#define VECTORSEARCHALGORITHM_HPP

#include <vector>
#include <utility> // For std::pair

template<typename T>
class VectorSearchAlgorithm {
public:
    virtual ~VectorSearchAlgorithm() {}

    // Searches for the closest vector to 'target'.
    // 'target' is the vector for which we are trying to find the closest match.
    // Returns a vector of std::pair. The first element of the pair is of type T,
    // representing some metric or distance, and the second element is the closest vector found of type std::vector<T>.
    // If no vectors are available for comparison, returns an empty vector.
    virtual std::vector<std::pair<T, std::vector<float>>> searchClosest (const std::vector<float>& target, const int ef = 1) = 0;
};

#endif // VECTORSEARCHALGORITHM_HPP
