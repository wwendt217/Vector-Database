#ifndef ANNOY_TREE_FOREST_HPP
#define ANNOY_TREE_FOREST_HPP

#include <vector>
#include <algorithm>
#include <limits>
#include <random>
#include <future>
#include <iterator>
#include <execution>

#include "VectorSearchAlgorithm.hpp"
#include "AnnoyTree.hpp"

template<typename TypeName>
class AnnoyTreeForest : public VectorSearchAlgorithm<TypeName> {
public:
    // Vector of unique pointers to AnnoyTrees
    std::vector<std::unique_ptr<AnnoyTree<TypeName>>> trees;

    float threshold;
    int sufficient_bucket_threshold;
    int max_depth;
    int n_trees;
    bool build_parallel;
    int vector_len;

    // Constructor that takes dataset and builds each tree in the forest
    AnnoyTreeForest(const std::vector<std::pair<TypeName, std::vector<float>>>& data,
                    int vector_len, 
                    float threshold,
                    int sufficient_bucket_threshold,
                    int max_depth,
                    int n_trees,
                    bool build_parallel = false) : threshold(threshold),
                                                   sufficient_bucket_threshold(sufficient_bucket_threshold),
                                                   max_depth(max_depth),
                                                   n_trees(n_trees),
                                                   build_parallel(build_parallel),
                                                   vector_len(vector_len) {
        trees.reserve(n_trees);
        if (build_parallel) {
            // Build trees in parallel
            std::vector<std::future<void>> futures;
            for (int i = 0; i < n_trees; ++i) {
                futures.push_back(std::async(std::launch::async, [this, &data, threshold, sufficient_bucket_threshold, max_depth]() {
                    auto tree = std::make_unique<AnnoyTree<TypeName>>(data, threshold, sufficient_bucket_threshold, max_depth);
                    trees.push_back(std::move(tree));
                }));
            }
            // Wait for all futures to complete
            for (auto& fut : futures) {
                fut.wait();
            }
        } else {
            // Build trees sequentially
            for (int i = 0; i < n_trees; ++i) {
                trees.push_back(std::make_unique<AnnoyTree<TypeName>>(data, threshold, sufficient_bucket_threshold, max_depth ));
            }
        }
    }

    std::vector<std::pair<TypeName, std::vector<float>>> searchClosest (const std::vector<float>& target, const int ef = 1) override {
        // Perform the search to get a vector of shared pointers to nodes
        auto nodes = query(target, ef);
    
        // Prepare the result vector with the appropriate format
        std::vector<std::pair<TypeName, std::vector<float>>> results;
        results.reserve(nodes.size());

        // Transform each node into the required format: a pair of T (node value) and vector<float> (node vector)
        for (const auto& node : nodes) {
            // Calculate the distance for each node as per the new requirements
            // The result should include both the distance and the node vector, but based on the given interface
            // we should only return the node's value (of type T) and its vector.
            results.emplace_back(std::get<0>(node), std::get<2>(node));
        }

        return results; 
    }

    std::vector<std::tuple<TypeName, float, std::vector<float>>> query(const std::vector<float>& vec, int k) const {
        // Temporary storage for futures that will hold the results from each tree
        std::vector<std::future<std::vector<std::tuple<TypeName, float, std::vector<float>>>>> futures;

        // Temporary storage for results, including the TypeName, distance, and vector
        std::vector<std::tuple<TypeName, float, std::vector<float>>> tempResults;

        // Launch asynchronous tasks for each tree if build_parallel is true
        for (const auto& tree : trees) {
            auto task = [&tree, &vec, this]() -> std::vector<std::tuple<TypeName, float, std::vector<float>>> {
                std::vector<std::tuple<TypeName, float, std::vector<float>>> results;
                const auto list = tree->findContainingList(vec);
                for (const auto& item : list) {
                    float distance = tree->calculateSquaredDistance(vec, item.second);
                    results.emplace_back(item.first, distance, item.second);
                }
                return results;
            };

            if (build_parallel) {
                futures.push_back(std::async(std::launch::async, task));
            } else {
                // For sequential execution, directly collect the results
                auto results = task();
                std::copy(results.begin(), results.end(), std::back_inserter(tempResults));
            }
        }

        if (build_parallel) {
            // Collect results from all futures
            for (auto& fut : futures) {
                auto results = fut.get();
                std::copy(results.begin(), results.end(), std::back_inserter(tempResults));
            }
        }

        std::partial_sort(tempResults.begin(), std::min(tempResults.begin() + k, tempResults.end()), tempResults.end(), [](const auto& a, const auto& b) {
            return std::get<1>(a) < std::get<1>(b); // Sorting based on distance
        });

        // Prepare the final vector to return, selecting the top k items based on distance
        std::vector<std::tuple<TypeName, float, std::vector<float>>> nearestNeighbors;
        for (int i = 0; i < std::min(k, static_cast<int>(tempResults.size())); ++i) {
            nearestNeighbors.push_back(tempResults[i]);
        }

        return nearestNeighbors;
    }
};

#endif // ANNOY_TREE_FOREST_HPP
