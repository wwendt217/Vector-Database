#ifndef BINARYTREE_HPP
#define BINARYTREE_HPP

#include "TreeNode.hpp"
#include <iostream>
#include <functional> // For std::function

template<typename T>
struct BinaryTree {
    std::shared_ptr<TreeNode<T>> root;

    BinaryTree();

    // Modified insert method to accept a comparator function
    void insert(T data, std::function<bool(T, T)> comparator);
    void inOrderTraversal(std::function<void(T)> visit) const;
    void depthFirstSearch(std::function<void(T)> visit) const;
private:
    // Modified insertHelper to use the comparator
    void insertHelper(std::shared_ptr<TreeNode<T>>& node, T data, std::function<bool(T, T)> comparator);
    void inOrderTraversalHelper(std::shared_ptr<TreeNode<T>> node, std::function<void(T)> visit) const;
    void depthFirstSearchHelper(std::shared_ptr<TreeNode<T>> node, std::function<void(T)> visit) const;
};

template<typename T>
BinaryTree<T>::BinaryTree() : root(nullptr) {}

#endif // BINARYTREE_HPP
