#ifndef DIRECTED_GRAPH_NODE_HPP
#define DIRECTED_GRAPH_NODE_HPP

#include <iostream>
#include <vector>
#include <memory>

template<typename T>
class DirectedGraphNode {
public:
    T value;
    // Outgoing adjacency list
    std::vector<std::shared_ptr<DirectedGraphNode<T>>> outgoingAdjList;
    // Incoming adjacency list
    std::vector<std::shared_ptr<DirectedGraphNode<T>>> incomingAdjList;

    DirectedGraphNode(T val) : value(val) {}

    // Add an outgoing edge to another node
    void addOutgoingEdge(std::shared_ptr<DirectedGraphNode<T>> other) {
        outgoingAdjList.push_back(other);
    }

    // Add an incoming edge from another node
    void addIncomingEdge(std::shared_ptr<DirectedGraphNode<T>> other) {
        incomingAdjList.push_back(other);
    }

    // Print all outgoing adjacent nodes
    void printOutgoingAdjacents() const {
        std::cout << "Outgoing adjacency list of node " << value << ": ";
        for (const auto& node : outgoingAdjList) {
            std::cout << node->value << " ";
        }
        std::cout << std::endl;
    }

    // Print all incoming adjacent nodes
    void printIncomingAdjacents() const {
        std::cout << "Incoming adjacency list of node " << value << ": ";
        for (const auto& node : incomingAdjList) {
            std::cout << node->value << " ";
        }
        std::cout << std::endl;
    }
};

#endif // DIRECTED_GRAPH_NODE_HPP
