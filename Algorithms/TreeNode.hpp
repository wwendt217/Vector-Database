#ifndef TREENODE_HPP
#define TREENODE_HPP

#include <memory>
#include <utility>

// Define the structure for a tree node using templates
template<typename T>
struct TreeNode {
    T data; // The data of generic type T
    std::shared_ptr<TreeNode<T>> left; // Pointer to the left child
    std::shared_ptr<TreeNode<T>> right; // Pointer to the right child

    // Constructor
    TreeNode() : left(nullptr), right(nullptr) {}
};

#endif // TREENODE_HPP
