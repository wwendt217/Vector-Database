    #ifndef HNSW_GRAPH_HPP
    #define HNSW_GRAPH_HPP

    #include <vector>
    #include <memory>
    #include <utility>
    #include <limits>
    #include <cmath>
    #include <queue>
    #include <set>
    #include <algorithm>
    #include <cmath>
    #include <stdexcept>
    #include <unordered_set>
    #include <random>

    #include "GraphNode.hpp"
    #include "VectorSearchAlgorithm.hpp"
    #include "Distances.hpp"

    template<typename T>
    class HNSW_graph : public VectorSearchAlgorithm<T> {
    public:
        using NodeValueType = std::pair<T, std::vector<float>>;
        using Node = GraphNode<NodeValueType>;
        using GraphLayer = std::vector<std::shared_ptr<Node>>;
        float mL;

        std::vector<GraphLayer> layers;
        int vector_len;
        int num_layers;
        int efc;
        float (*distanceFunction)(const std::vector<float>&, const std::vector<float>&) = defaultDistance;

        HNSW_graph(const std::vector<NodeValueType>& nodeValues, 
                   float mL,
                   int vector_len,
                   int num_layers,
                   int efc,
                   float (*distanceFunction)(const std::vector<float>&, const std::vector<float>&) = defaultDistance) :  
                          layers (num_layers), 
                          mL (mL),
                          vector_len (vector_len),
                          num_layers (num_layers),
                          efc (efc),
                          distanceFunction (distanceFunction) {
            // Create a copy of the input vector
            std::vector<NodeValueType> shuffledValues = nodeValues;

            // Shuffle the copied vector 
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(shuffledValues.begin(), shuffledValues.end(), g);

            // Insert each value into the graph
            for (const auto& value : shuffledValues) {
                insert(value);
            }
        }

        // Default constructor
        HNSW_graph() : mL(0.9f) { 
            layers = std::vector<GraphLayer>(num_layers); // Initialize layers based on the num_layers template argument
        }

        std::vector<std::shared_ptr<Node>> search_layer(int layerIndex, const std::shared_ptr<Node>& startNode, const std::vector<float>& queryVec, size_t ef = 1) {
            if (layerIndex < 0 || layerIndex >= num_layers) throw std::out_of_range("Layer index is out of range.");

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

            float initialDistance = distanceFunction(startNode->value.second, queryVec);
            candidates.emplace(initialDistance, startNode);
            nearest_neighbors.emplace(initialDistance, startNode);
            visited_nodes.insert(startNode);

            while (!candidates.empty()) {
                auto current = candidates.top();
                candidates.pop();

                // If nearest_neighbors is full and current candidate is not closer, break
                if (nearest_neighbors.size() > 0 && current.first > nearest_neighbors.top().first) {
                    break;
                }

                // Continue searching through adjacents
                for (auto neighbor : current.second->adjacentsByGraph[layerIndex]) {
                    if (visited_nodes.insert(neighbor).second) { // Node wasn't visited before
                        float distance = distanceFunction(neighbor->value.second, queryVec);
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

        std::vector<std::pair<T, std::vector<float>>> searchClosest (const std::vector<float>& target, const int ef = 1) override {
            auto nodes = search (target, ef);

            // Prepare the result vector with the appropriate format
            std::vector<std::pair<T, std::vector<float>>> results;
            results.reserve(nodes.size());

            // Transform each node into the required format: a pair of T (node value) and vector<float> (node vector)
            for (const auto& node : nodes) {
                results.emplace_back(node->value.first, node->value.second);
            }

            return results; 
        }

        std::vector<std::shared_ptr<Node>> search(const std::vector<float>& queryVec, size_t ef = 1) {
            std::vector<std::shared_ptr<Node>> result;
            if (layers.empty() || layers[0].empty()) {
                return result;
            }

            auto bestNode = layers[0][0];
            for (int i = 0; i < num_layers; ++i) {
                bestNode = search_layer(i, bestNode, queryVec)[0];
            }
            auto results = search_layer(num_layers - 1, bestNode, queryVec, ef);
            for (const auto& r : results) {
                result.push_back(r);
            }

            return result;
        }

        void insert(NodeValueType value) {
            std::shared_ptr<Node> new_node = std::make_shared<Node>(value);
            
            if (layers[0].empty()) {
                for(auto& layer : layers) {
                    layer.push_back(new_node);
                }
                return;
            }

            int insertion_layer = calculate_insertion_layer();
            auto& curr_node = layers[0][0];
            for (int i = 0; i < num_layers; i++) {
                if (i < insertion_layer) {
                    curr_node = search_layer(i, curr_node, value.second)[0];
                } else {
                    auto nearest_neighbors = search_layer (i, curr_node, value.second, efc);
                    for (auto& neighbor : nearest_neighbors) {
                        connectNodesInLayer(neighbor, new_node, i);
                    }
                    curr_node = nearest_neighbors[0]; 
                    //layers[i].push_back(new_node);
                }
            }
        }



        // Additional functionalities...
        // Method to add a new node to a specific layer
        void addNodeToLayer(const T& value, const std::vector<float>& vector, int layerIndex) {
            if (layerIndex < 0 || layerIndex >= num_layers) {
                throw std::out_of_range("Layer index is out of range.");
            }
            auto newNode = std::make_shared<Node>(NodeValueType(value, vector));
            layers[layerIndex].push_back(newNode);
        }

        // Method to connect two nodes within a specific layer
        void connectNodesInLayer(std::shared_ptr<Node> node1, std::shared_ptr<Node> node2, int layerIndex) {
            if (layerIndex < 0 || layerIndex >= num_layers || !node1 || !node2) {
                throw std::invalid_argument("Invalid layer index or node pointer.");
            }
            
            node1->addEdge(layerIndex, node2);
            node2->addEdge(layerIndex, node1);
        }

    private:
        int calculate_insertion_layer() {
            // mL is a multiplicative factor used to normalize the distribution
            int l = -static_cast<int>(std::log(static_cast<double>(std::rand()) / RAND_MAX) * mL);
            return std::min(l, num_layers - 1);
        }
    };

    #endif // HNSW_GRAPH_HPP
