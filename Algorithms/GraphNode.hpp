#include <iostream>
#include <map>
#include <string>
#include <vector>

template<typename T>
class GraphNode {
public:
    T value;
    // Each graph is identified by an integer (or string) and has its own adjacency list.
    std::map<int, std::vector<std::shared_ptr<GraphNode<T>>>> adjacentsByGraph;

    GraphNode(T val) : value(val) {}

    // Add an edge in a specific graph
    void addEdge(int graphId, std::shared_ptr<GraphNode<T>> other) {
        adjacentsByGraph[graphId].push_back(other);
    }

    // Print all adjacent nodes for a specific graph
    void printAdjacents(int graphId) const {
        std::cout << "Node " << value << " in Graph " << graphId << " is connected to: ";
        if (adjacentsByGraph.find(graphId) != adjacentsByGraph.end()) {
            for (const auto& node : adjacentsByGraph.at(graphId)) {
                std::cout << node->value << " ";
            }
        }
        std::cout << std::endl;
    }
};