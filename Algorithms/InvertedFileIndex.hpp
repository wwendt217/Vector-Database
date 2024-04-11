#ifndef INVERTEDFILEINDEX_HPP
#define INVERTEDFILEINDEX_HPP

#include <vector>
#include <unordered_map>
#include <cmath>
#include <limits>
#include <algorithm>
#include <random>
#include <numeric>
#include <stdexcept>

#include "VectorSearchAlgorithm.hpp"

template<typename T>
class InvertedFileIndex : public VectorSearchAlgorithm<T> {
public:
    int vector_len;
    int num_centroids; 
    int retrain_threshold;

    // Constructor initializes and trains the model on the initial dataset
    InvertedFileIndex(const std::vector<std::pair<T, std::vector<float>>>& inputData,
                      int vector_len, 
                      int num_centroids, 
                      int retrain_threshold = 1
    ) : data(inputData), 
        clusters(num_centroids),
        vector_len(vector_len),
        num_centroids(num_centroids),
        retrain_threshold(retrain_threshold) {
        if (data.size() < static_cast<size_t>(num_centroids)) {
            throw std::invalid_argument("Data size must be larger than the number of centroids.");
        }
        initializeCentroids();
        retrain();
    }

    // Adds a new data point and retrains the model if necessary
    void add(const T& id, const std::vector<float>& vec) {
        if (vec.size() != static_cast<size_t>(vector_len)) {
            throw std::invalid_argument("Vector length does not match the specified vector_len.");
        }
        data.push_back({id, vec});
        if (++nodesAddedSinceLastRetrain >= retrain_threshold) {
            retrain();
            nodesAddedSinceLastRetrain = 0;
        }
    }

    std::vector<std::pair<T, std::vector<float>>> searchClosest(const std::vector<float>& vec, int num_results) override {
        return findClosest(vec, num_results);
    }

    // Finds the num_results closest vectors to the input vector
    std::vector<std::pair<T, std::vector<float>>> findClosest(const std::vector<float>& vec, int num_results) {
        std::vector<std::pair<float, T>> distances; // Pair of distance and ID

        for (const auto& item : data) {
            float distance = euclideanDistance(item.second, vec);
            distances.emplace_back(distance, item.first);
        }

        // Sort by distance
        std::nth_element(distances.begin(), distances.begin() + num_results, distances.end());
        distances.resize(num_results);

        // Retrieve the corresponding data points
        std::vector<std::pair<T, std::vector<float>>> results;
        for (const auto& dist : distances) {
            auto it = std::find_if(data.begin(), data.end(), [&dist](const auto& item) {
                return item.first == dist.second;
            });
            if (it != data.end()) {
                results.push_back(*it);
            }
        }

        return results;
    }

private:
    std::vector<std::pair<T, std::vector<float>>> data;
    std::vector<std::vector<float>> centroids;
    std::vector<std::vector<std::pair<T, std::vector<float>>>> clusters;
    int nodesAddedSinceLastRetrain = 0;

    // Initializes centroids by randomly selecting data points
    void initializeCentroids() {
        std::vector<int> indices(data.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device{}()));

        centroids.clear();
        for (int i = 0; i < num_centroids; ++i) {
            centroids.push_back(data[indices[i]].second);
        }
    }

    // The core retraining function
    void retrain() {
        bool centroidsChanged;
        do {
            centroidsChanged = assignToNearestCentroids();
            centroidsChanged = updateCentroids() || centroidsChanged;
        } while (centroidsChanged);
    }

    static constexpr float convergenceThreshold = 0.001f; // Minimum movement of centroids to continue training
    // Assigns each data point to the nearest centroid
    bool assignToNearestCentroids() {
        bool centroidsChanged = false;
        std::vector<std::vector<std::pair<T, std::vector<float>>>> newClusters(num_centroids);
        
        for (const auto& item : data) {
            int nearestCentroidIndex = findNearestCentroid(item.second);
            newClusters[nearestCentroidIndex].push_back(item);
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

    // Updates centroids based on current cluster assignments
    bool updateCentroids() {
        bool anyCentroidMoved = false;
        std::vector<std::vector<float>> newCentroids(num_centroids, std::vector<float>(vector_len, 0.0));
        
        // Accumulate all vectors assigned to each centroid
        for (int i = 0; i < num_centroids; ++i) {
            if (!clusters[i].empty()) {
                for (const auto& pair : clusters[i]) {
                    for (int j = 0; j < vector_len; ++j) {
                        newCentroids[i][j] += pair.second[j];
                    }
                }
                for (int j = 0; j < vector_len; ++j) {
                    newCentroids[i][j] /= clusters[i].size();
                }
            }

            // Check if the centroid has moved significantly
            if (euclideanDistance(centroids[i], newCentroids[i]) >= convergenceThreshold) {
                anyCentroidMoved = true;
                centroids[i] = newCentroids[i];
            }
        }

        return anyCentroidMoved;
    }

    // Finds the index of the nearest centroid to a given vector
    int findNearestCentroid(const std::vector<float>& vec) {
        float minDistance = std::numeric_limits<float>::max();
        int nearestIndex = -1;
        for (int i = 0; i < num_centroids; ++i) {
            float distance = euclideanDistance(vec, centroids[i]);
            if (distance < minDistance) {
                minDistance = distance;
                nearestIndex = i;
            }
        }
        return nearestIndex;
    }

    float euclideanDistance(const std::vector<float>& vec1, const std::vector<float>& vec2) {
        float sum = 0.0;
        for (int i = 0; i < vector_len; ++i) {
            float diff = vec1[i] - vec2[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
};

#endif // INVERTEDFILEINDEX_HPP