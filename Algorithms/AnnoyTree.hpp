#ifndef ANNOY_TREE_HPP
#define ANNOY_TREE_HPP

#include <vector>
#include <functional>
#include <random>
#include <memory>
#include <queue>
#include <stack>
#include <stdexcept>

#include "BinaryTree.hpp"
#include "AnnoyTreeNodeData.hpp"

// How many times we should check if vec[idx1] != vec[idx2] before giving up
#define NUM_RANDOM_VECTORS_TO_TRY (5)

template<typename TypeName>
class AnnoyTree {
public:
    BinaryTree<AnnoyTreeNodeData<TypeName>> tree;

    int vector_len;
    float threshold;
    int sufficient_bucket_threshold;
    int max_depth;

    AnnoyTree(const std::vector<std::pair<TypeName, std::vector<float>>>& data, 
              float threshold, 
              int sufficient_bucket_threshold, 
              int max_depth ) : 
              threshold(threshold),
              sufficient_bucket_threshold(sufficient_bucket_threshold),
              max_depth(max_depth) {
        if (!data.empty()) {
            from_data(data);
        }
    }

    // Method to find the node's list that should contain the given vector
    std::vector<std::pair<TypeName, std::vector<float>>> findContainingList(const std::vector<float>& vec) const {
        //std::shared_ptr<TreeNode<AnnoyTreeNodeData<vector_len, TypeName>>> node = tree.root;
        
        std::vector<std::pair<TypeName, std::vector<float>>> results;

        std::queue<std::shared_ptr<TreeNode<AnnoyTreeNodeData<TypeName>>>> unprocessed_nodes;
        unprocessed_nodes.push(tree.root);

        while (!unprocessed_nodes.empty()) {
            auto& node = unprocessed_nodes.front();
            unprocessed_nodes.pop();
            // Assuming leaf nodes or the logic to determine if a node is a leaf
            // For simplicity, we check if the node has no children
            if (node->left == nullptr && node->right == nullptr) {
                results.insert(results.end(), node->data.pairList.begin(), node->data.pairList.end());
            } else {
                float distanceToVec1 = calculateSquaredDistance(vec, node->data.vec1);
                float distanceToVec2 = calculateSquaredDistance(vec, node->data.vec2);
                if (std::abs(distanceToVec1 - distanceToVec2) < threshold) {
                    unprocessed_nodes.push(node->left);
                    unprocessed_nodes.push(node->right);
                } else if (distanceToVec1 < distanceToVec2) {
                    unprocessed_nodes.push(node->left);
                } else {
                    unprocessed_nodes.push(node->right);
                }
            }
        }
        return results;
    }

    // Method to reconstruct the dataset from the BinaryTree
    std::vector<std::pair<TypeName, std::vector<float>>> reconstructData() const {
        std::vector<std::pair<TypeName, std::vector<float>>> dataset;
        reconstructDataHelper(tree.root, dataset);
        return dataset;
    }

    const float calculateSquaredDistance(const std::vector<float>& vec1, const std::vector<float>& vec2) const {
        float squaredDistance = 0.0f;
        for (size_t i = 0; i < vec1.size(); ++i) {
            squaredDistance += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
        }
        return squaredDistance;
    }

private:
    void reconstructDataHelper(const std::shared_ptr<TreeNode<AnnoyTreeNodeData<TypeName>>>& node, std::vector<std::pair<TypeName, std::vector<float>>>& dataset) const {
        if (!node) return;
        if (!node->left && !node->right) {
            dataset.insert(dataset.end(), node->data.pairList.begin(), node->data.pairList.end());
        } else {
            reconstructDataHelper(node->left, dataset);
            reconstructDataHelper(node->right, dataset);
        }
    }
    /// @brief 
    /// @param data 
    void from_data(const std::vector<std::pair<TypeName, std::vector<float>>>& data) {
        std::stack<std::pair<std::shared_ptr<TreeNode<AnnoyTreeNodeData<TypeName>>>, std::vector<std::pair<TypeName, std::vector<float>>>>> tasks;
        std::stack<int> depths;

        tree.root = std::make_shared<TreeNode<AnnoyTreeNodeData<TypeName>>>();
        tasks.push({tree.root, data}); // No move here; data is a const reference

        depths.push(0);

        while (!tasks.empty()) {
            auto [node, currentData] = tasks.top();
            tasks.pop();

            int depth = depths.top();
            depths.pop();

            if (currentData.size() <= sufficient_bucket_threshold || depth > max_depth) {
                // Assuming TreeNode has a way to store or otherwise handle the currentData directly
                node->data.pairList.assign(std::make_move_iterator(currentData.begin()), std::make_move_iterator(currentData.end()));
                continue;
            }

            auto selectedVectors = selectRandomVectors(currentData);
            auto [leftData, rightData] = splitData(std::move(currentData), selectedVectors);

            // Now, leftData and rightData are moved into the tasks, not copied
            if (!leftData.empty()) {
                node->left = std::make_shared<TreeNode<AnnoyTreeNodeData<TypeName>>>();
                tasks.push({node->left, std::move(leftData)});
                depths.push(depth + 1);
            }
            if (!rightData.empty()) {
                node->right = std::make_shared<TreeNode<AnnoyTreeNodeData<TypeName>>>();
                tasks.push({node->right, std::move(rightData)});
                depths.push(depth + 1);
            }
        }
    }

    std::pair<std::vector<float>, std::vector<float>> selectRandomVectors(const std::vector<std::pair<TypeName, std::vector<float>>>& data) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, data.size() - 1);
        
        int idx1 = dis(gen), idx2 = dis(gen);
        for (int i = 0; i < NUM_RANDOM_VECTORS_TO_TRY && idx1 == idx2; ++i) {
            idx2 = dis(gen);
        }

        return {data[idx1].second, data[idx2].second};
    }

    // Assumes the original data can be consumed/modified, thus passed by value
    // Assumes <random> is included
    std::pair<std::vector<std::pair<TypeName, std::vector<float>>>, std::vector<std::pair<TypeName, std::vector<float>>>>
    splitData(std::vector<std::pair<TypeName, std::vector<float>>> data, const std::pair<std::vector<float>, std::vector<float>>& vectors) {
        std::vector<std::pair<TypeName, std::vector<float>>> leftData, rightData;
    
        if (vectors.first == vectors.second) { // Check if vec1 and vec2 are the same
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, 1); // For randomly splitting

            for (auto& item : data) {
                if (dis(gen) == 0) {
                    leftData.push_back(std::move(item));
                } else {
                    rightData.push_back(std::move(item));
                }
            }
        } else {
            // Original splitting logic based on distance
            for (auto& item : data) {
                float distanceToFirstSquared = calculateSquaredDistance(item.second, vectors.first);
                float distanceToSecondSquared = calculateSquaredDistance(item.second, vectors.second);

                if (distanceToFirstSquared < distanceToSecondSquared) {
                    leftData.push_back(std::move(item));
                } else {
                    rightData.push_back(std::move(item));
                }
            }
        }

        return {std::move(leftData), std::move(rightData)};
    }

};

#endif // ANNOY_TREE_HPP