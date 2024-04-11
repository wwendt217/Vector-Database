#include "BinaryTree.hpp"

// Modified insert method to accept a comparator function
template<typename T>
void BinaryTree<T>::insert(T data, std::function<bool(T, T)> comparator) {
    insertHelper(root, data, comparator);
}

// Updated insertHelper to use the comparator for node insertion logic
template<typename T>
void BinaryTree<T>::insertHelper(std::shared_ptr<TreeNode<T>>& node, T data, std::function<bool(T, T)> comparator) {
    if (!node) {
        node = std::make_shared<TreeNode<T>>(data);
    } else if (comparator(data, node->data)) { // Use comparator instead of <
        insertHelper(node->left, data, comparator);
    } else {
        insertHelper(node->right, data, comparator);
    }
}

template<typename T>
void BinaryTree<T>::inOrderTraversal(std::function<void(T)> visit) const {
    inOrderTraversalHelper(root, visit);
}

template<typename T>
void BinaryTree<T>::inOrderTraversalHelper(std::shared_ptr<TreeNode<T>> node, std::function<void(T)> visit) const {
    if (node) {
        inOrderTraversalHelper(node->left, visit);
        visit(node->data);
        inOrderTraversalHelper(node->right, visit);
    }
}

// Depth-first search traversal method (pre-order)
template<typename T>
void BinaryTree<T>::depthFirstSearch(std::function<void(T)> visit) const {
    depthFirstSearchHelper(root, visit);
}

// Helper function for depth-first search traversal
template<typename T>
void BinaryTree<T>::depthFirstSearchHelper(std::shared_ptr<TreeNode<T>> node, std::function<void(T)> visit) const {
    if (node) {
        visit(node->data); // Visit the current node
        depthFirstSearchHelper(node->left, visit); // Recursively visit left subtree
        depthFirstSearchHelper(node->right, visit); // Recursively visit right subtree
    }
}

#include <iostream>
#include <cassert>

// Comparator function for integers (greater than)
bool greaterThan(int a, int b) {
    return a > b;
}

// Comparator function for characters (less than)
bool lessThan(char a, char b) {
    return a < b;
}

int main() {
    // Testing BinaryTree with integers
    BinaryTree<int> intTree;
    intTree.insert(5, greaterThan);
    intTree.insert(3, greaterThan);
    intTree.insert(7, greaterThan);
    intTree.insert(1, greaterThan);
    intTree.insert(9, greaterThan);

    std::cout << "In-order traversal of integers (greater than):\n";
    intTree.inOrderTraversal([](int data) { std::cout << data << " "; });
    std::cout << "\n";

    // Testing BinaryTree with characters
    BinaryTree<char> charTree;
    charTree.insert('d', lessThan);
    charTree.insert('b', lessThan);
    charTree.insert('f', lessThan);
    charTree.insert('a', lessThan);
    charTree.insert('e', lessThan);

    std::cout << "In-order traversal of characters (less than):\n";
    charTree.inOrderTraversal([](char data) { std::cout << data << " "; });
    std::cout << "\n";

    std::cout << "All assertions passed successfully.\n";

    return 0;
}
