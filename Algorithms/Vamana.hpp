#ifndef VAMANA_HPP
#define VAMANA_HPP

#include <iostream>
#include <vector>
#include <memory>
#include <queue>
#include <set>
#include <random>
#include <unordered_set>
#include <functional>

#include "VectorSearchAlgorithm.hpp"
#include "DirectedGraphNode.hpp"
#include "Distances.hpp"

template<typename T>
class Vamana : public VectorSearchAlgorithm<T> {
public:
    using NodeValueType = std::pair<T, std::vector<float>>;
    using Node = DirectedGraphNode<NodeValueType>;

    std::vector<std::shared_ptr<Node>> nodes;
    float alpha;
    int vector_len; 
    int R;
    int nq;
    float (*distanceFunction)(const std::vector<float>&, const std::vector<float>&);

    Vamana(const std::vector<std::pair<T, std::vector<float>>>& nodeValues, 
           float alpha,
           int vector_len,
           int R,
           int nq = 1,
           float (*distanceFunction)(const std::vector<float>&, const std::vector<float>&) = defaultDistance) 
           : alpha(alpha),
             vector_len(vector_len),
             R(R),
             nq(nq),
             distanceFunction(distanceFunction) {
        build_rng(nodeValues);
    }

    std::vector<std::pair<T, std::vector<float>>> searchClosest(const std::vector<float>& target, const int ef = 1) override {
        // Perform the search to get a vector of shared pointers to nodes
        auto nodes = search(target, ef);
    
        // Prepare the result vector with the appropriate format
        std::vector<std::pair<T, std::vector<float>>> results;
        results.reserve(nodes.size());

        // Transform each node into the required format: a pair of T (node value) and vector<float> (node vector)
        for (const auto& node : nodes) {
            results.emplace_back(node->value.first, node->value.second);
        }

        return results; 
    }

    // Function to build the random neighborhood graph
    void build_rng(const std::vector<std::pair<T, std::vector<float>>>& nodeValues) {
        for (const auto& value : nodeValues) {
            addNode(value);
        }

        find_start_node();

        // Randomize edges
        for (auto& node : nodes) {
            // Shuffle the nodes vector (excluding the current node)
            auto start = nodes.begin();
            auto end = nodes.end();
            std::shuffle(start, end, std::default_random_engine(std::random_device()()));

            // Add outgoing edges from the current node to the first R nodes (excluding itself)
            int count = 0;
            for (auto it = start; it != end && count < R; ++it) {
                if (*it != node) {
                    connectNodes(node, *it);
                    ++count;
                }
            }
        }

        // Robust prune
        for (auto& node : nodes) {
            auto V = search(node->value.second);
            robust_prune(node, V);
            for (auto& inbound_node : node->incomingAdjList) {
                if (inbound_node->incomingAdjList.size() > R) {
                    std::vector<std::shared_ptr<Node>> list = inbound_node->incomingAdjList;
                    list.push_back(node);
                    robust_prune(inbound_node, list);
                } else {
                    inbound_node->addIncomingEdge(node);
                }
            }
        }
    }

    void robust_prune(std::shared_ptr<Node>& node, const std::vector<std::shared_ptr<Node>>& V) {
        // Combine V with the outgoing adjacency list of the node
        std::vector<std::shared_ptr<Node>> combinedList(V.size() + node->outgoingAdjList.size());
        auto it = std::set_union(V.begin(), V.end(), node->outgoingAdjList.begin(), node->outgoingAdjList.end(), combinedList.begin());
        combinedList.resize(it - combinedList.begin());

        // Remove the node from the combined list if it exists
        combinedList.erase(std::remove(combinedList.begin(), combinedList.end(), node), combinedList.end());

        // Erase the node from its own outgoing adjacency list
        node->outgoingAdjList.erase(std::remove_if(node->outgoingAdjList.begin(), node->outgoingAdjList.end(),
                                                    [&](const auto& n) { return n == node; }),
                                     node->outgoingAdjList.end());

        // Create a set (min heap) out of V using the distance from the values of V to node
        auto minCompare = [&](const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
            float distanceA = defaultDistance(a->value.second, node->value.second);
            float distanceB = defaultDistance(b->value.second, node->value.second);
            return distanceA > distanceB; // Invert the comparison for min heap
        };
        std::set<std::shared_ptr<Node>, decltype(minCompare)> minHeap(minCompare);

        for (const auto& n : combinedList) {
            minHeap.insert(n);
        }

        // Other pruning logic here
        while (!minHeap.empty()) {
            // Process the top node in the min heap
            auto topNode = *minHeap.begin();
            minHeap.erase(minHeap.begin());

            // Add topNode to the outgoing edges of node
            node->addOutgoingEdge(topNode);

            // Break the loop and return if the size of outgoing edges of node reaches R
            if (node->outgoingAdjList.size() == R) {
                break;
            }

            // Make a copy of the heap
            auto copyHeap = minHeap;

            while (!copyHeap.empty()) {
                auto elem = *copyHeap.begin();
                copyHeap.erase(copyHeap.begin());
                if (alpha * defaultDistance(elem->value.second, node->value.second) <= defaultDistance(topNode->value.second, node->value.second)) {
                    minHeap.erase(elem);
                }
            }
        }
    }


    // Greedy search function
    std::vector<std::shared_ptr<Node>> search(const std::vector<float>& queryVec, size_t ef = 1) {

            std::unordered_set<std::shared_ptr<Node>> visited_nodes;
        
            // Max heap for nearest_neighbors based on distance
            auto maxCompare = [](const std::pair<float, std::shared_ptr<Node>>& a, const std::pair<float, std::shared_ptr<Node>>& b) {
                return a.first < b.first;
            };
            std::priority_queue<std::pair<float, std::shared_ptr<Node>>, 
                                std::vector<std::pair<float, std::shared_ptr<Node>>>, 
                                decltype(maxCompare)> nearest_neighbors(maxCompare);

            // Min heap for candidates based on distance
            auto minCompare = [](const std::pair<float, std::shared_ptr<Node>>& a, const std::pair<float, std::shared_ptr<Node>>& b) {
                return a.first > b.first;
            };
            std::priority_queue<std::pair<float, std::shared_ptr<Node>>, 
                                std::vector<std::pair<float, std::shared_ptr<Node>>>, 
                                decltype(minCompare)> candidates(minCompare);

            float initialDistance = defaultDistance(startNode->value.second, queryVec);
            candidates.emplace(initialDistance, startNode);
            nearest_neighbors.emplace(initialDistance, startNode);
            visited_nodes.insert(startNode);

            while (!candidates.empty()) {
                auto current = candidates.top();
                candidates.pop();

                // If nearest_neighbors is full and current candidate is not closer, break
                if (nearest_neighbors.size() > ef - 1 && current.first > nearest_neighbors.top().first) {
                    break;
                }

                // Continue searching through adjacents
                for (auto neighbor : current.second->outgoingAdjList) {
                    if (visited_nodes.insert(neighbor).second) { // Node wasn't visited before
                        float distance = defaultDistance(neighbor->value.second, queryVec);
                        if (distance < nearest_neighbors.top().first || nearest_neighbors.size() < ef) {
                            candidates.push({distance, neighbor});
                            nearest_neighbors.push({distance, neighbor});
                            if (nearest_neighbors.size() > ef) {
                                nearest_neighbors.pop();
                            }
                        }
                    }
                }
            }

            // Transfer from max heap to vector without reversing
            std::vector<std::shared_ptr<Node>> result;
            while (!nearest_neighbors.empty()) {
                result.insert(result.begin(), nearest_neighbors.top().second); // Insert at the beginning
                nearest_neighbors.pop();
            }

            return result;
    }

private:
    // Function to find the start node closest to the average of all vectors in nodeValues
    void find_start_node() {
        if (nodes.empty()) {
            return;
        }

        std::vector<float> averageVector(vector_len, 0.0f);
        for (const auto& node : nodes) {
            for (size_t i = 0; i < node->value.second.size(); ++i) {
                averageVector[i] += node->value.second[i];
            }
        }

        for (size_t i = 0; i < averageVector.size(); ++i) {
            averageVector[i] /= nodes.size();
        }

        float minDistance = std::numeric_limits<float>::max();
        std::shared_ptr<Node> closestNode = nullptr;

        for (const auto& node : nodes) {
            float distance = defaultDistance(node->value.second, averageVector);
            if (distance < minDistance) {
                minDistance = distance;
                closestNode = node;
            }
        }

        startNode = closestNode;
    }

    std::shared_ptr<Node> startNode;

    // Method to add a new node
    void addNode(const NodeValueType& value) {
        auto newNode = std::make_shared<DirectedGraphNode<NodeValueType>>(value); // Using DirectedGraphNode
        nodes.push_back(newNode);
    }

    // Method to connect two nodes
    void connectNodes(std::shared_ptr<DirectedGraphNode<NodeValueType>> node1, std::shared_ptr<DirectedGraphNode<NodeValueType>> node2) {
        if (!node1 || !node2) {
            throw std::invalid_argument("Invalid node pointer.");
        }
        // Assuming both nodes are already part of the graph
        node1->addOutgoingEdge(node2);
        node2->addIncomingEdge(node1);
    }
};



#endif // VAMANA_HPP