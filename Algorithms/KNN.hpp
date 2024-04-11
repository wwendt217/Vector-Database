#ifndef KNN_HPP
#define KNN_HPP

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <random>
#include <numeric>
#include <functional>

// Default squared Euclidean distance function
float defaultSquaredDistance(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum; // Squared distance, without taking square root
}

template<int vector_len, int num_centroids>
class KNN {
public:
    using Vector = std::vector<float>;
    using DataSet = std::vector<Vector>;
    // Using std::function for distance function to allow capturing vector_len if needed
    using DistanceFunction = std::function<float(const Vector&, const Vector&)>;

private:
    DataSet centroids;
    std::vector<DataSet> clusters; // Cluster assignments
    static constexpr float convergenceThreshold = 0.001f;
    DistanceFunction distanceFunction;

public:
    KNN(DistanceFunction distFunc = defaultSquaredDistance)
    : clusters(num_centroids), distanceFunction(distFunc) {
        centroids.resize(num_centroids, Vector(vector_len, 0.0f));
    }

    void train(const DataSet& data) {
        initializeCentroids(data);
        bool centroidsChanged;
        do {
            centroidsChanged = assignToNearestCentroids(data);
            centroidsChanged = updateCentroids() || centroidsChanged;
        } while (centroidsChanged);
    }

    int predict(const Vector& vec) const {
        int nearestIndex = 0;
        float minDistance = std::numeric_limits<float>::max();
        for (int i = 0; i < num_centroids; ++i) {
            float distance = distanceFunction(vec, centroids[i]);
            if (distance < minDistance) {
                minDistance = distance;
                nearestIndex = i;
            }
        }
        return nearestIndex;
    }

private:
    void initializeCentroids(const DataSet& data) {
        std::vector<int> indices(data.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device{}()));

        for (int i = 0; i < num_centroids; ++i) {
            centroids[i] = data[indices[i] % data.size()];
        }
    }

    bool assignToNearestCentroids(const DataSet& data) {
        bool centroidsChanged = false;
        std::vector<std::vector<std::vector<float>>> newClusters(num_centroids);
        
        for (const auto& vec : data) {
            int nearestCentroidIndex = predict(vec);
            newClusters[nearestCentroidIndex].push_back(vec);
        }

        // Compare newClusters with clusters to determine if centroids have changed
        for (int i = 0; i < num_centroids; ++i) {
            if (newClusters[i].size() != clusters[i].size()) {
                centroidsChanged = true;
                break;
            }
        }

        clusters = std::move(newClusters); // Update the clusters for the next iteration

        return centroidsChanged;
    }

    bool updateCentroids() {
        bool anyCentroidMoved = false;
        for (int i = 0; i < num_centroids; ++i) {
            Vector newCentroid(vector_len, 0.0f);
            if (!clusters[i].empty()) {
                for (const auto& vec : clusters[i]) {
                    for (int j = 0; j < vector_len; ++j) {
                        newCentroid[j] += vec[j];
                    }
                }
                for (int j = 0; j < vector_len; ++j) {
                    newCentroid[j] /= clusters[i].size();
                }

                if (euclideanDistance(centroids[i], newCentroid) >= convergenceThreshold) {
                    centroids[i] = newCentroid;
                    anyCentroidMoved = true;
                }
            }
        }
        return anyCentroidMoved;
    }

    // This uses the original euclideanDistance for internal centroid movement checks
    static float euclideanDistance(const Vector& a, const Vector& b) {
        float sum = 0.0f;
        for (int i = 0; i < vector_len; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
};

#endif // KNN_HPP
