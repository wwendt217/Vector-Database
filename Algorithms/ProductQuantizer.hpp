#ifndef PRODUCTQUANTIZER_HPP
#define PRODUCTQUANTIZER_HPP

#include "KNN.hpp"
#include <vector>
#include <functional>

template<int vector_len, int num_centroids, int num_subspaces>
class ProductQuantizer {
public:
    using Vector = std::vector<float>;
    using ProjectionFunction = std::function<Vector(const Vector&)>;
    using SubspaceKNN = KNN<vector_len / num_subspaces, num_centroids>;

private:
    std::vector<ProjectionFunction> projections; // Projection functions for each subspace
    std::vector<SubspaceKNN> knnModels; // A KNN model for each subspace

public:
    // Constructor takes a list of projection functions, one for each subspace
    ProductQuantizer(const std::vector<ProjectionFunction>& projections)
        : projections(projections) {
        if (projections.size() != num_subspaces) {
            throw std::invalid_argument("Number of projections must match num_subspaces.");
        }
        knnModels.resize(num_subspaces);
    }

    // Train each KNN model with projected data
    void train(const std::vector<Vector>& data) {
        for (size_t i = 0; i < projections.size(); ++i) {
            std::vector<Vector> projectedData;
            for (const auto& vec : data) {
                projectedData.push_back(projections[i](vec));
            }
            knnModels[i].train(projectedData);
        }
    }

    // Quantize a vector, returning the index of the nearest centroid in each subspace
    std::vector<int> quantize(const Vector& vec) const {
        std::vector<int> indices(num_subspaces);
        for (size_t i = 0; i < projections.size(); ++i) {
            Vector projectedVec = projections[i](vec);
            indices[i] = knnModels[i].predict(projectedVec);
        }
        return indices;
    }
};

#endif // PRODUCTQUANTIZER_HPP
