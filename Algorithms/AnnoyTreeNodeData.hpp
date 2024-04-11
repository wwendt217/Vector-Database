#ifndef ANNOY_TREE_NODE_DATA_HPP
#define ANNOY_TREE_NODE_DATA_HPP

#include <vector>
#include <utility>

template<typename DataType>
struct AnnoyTreeNodeData {
    std::vector<float> vec1;
    std::vector<float> vec2;
    std::vector<std::pair<DataType, std::vector<float>>> pairList;

    int vector_len;

    AnnoyTreeNodeData () : vector_len(1), vec1(1, 0.0f), vec2(1, 0.0f) {}

    // Constructor
    AnnoyTreeNodeData(int vector_length) : vector_len(vector_length), vec1(vector_len, 0.0f), vec2(vector_len, 0.0f) {
        // Optionally initialize vec1 and vec2 with specific values
        // and add initial data to pairList if necessary
    }

    // Another constructor that initializes vec1 and vec2 with specific values
    AnnoyTreeNodeData(const std::vector<float>& initialVec1, const std::vector<float>& initialVec2)
        : vec1(initialVec1), vec2(initialVec2) {
        // Ensure initialVec1 and initialVec2 have the correct length (vector_len)
        if (vec1.size() != vector_len || vec2.size() != vector_len) {
            throw std::invalid_argument("Vectors must have the length specified by vector_len");
        }
        // pairList can be populated later or modified to accept initial data
    }

    // Method to add data to pairList, ensuring vector length matches vector_len
    void addData(const DataType& data, const std::vector<float>& vec) {
        if (vec.size() == vector_len) {
            pairList.emplace_back(data, vec);
        } else {
            // Handle the error or ignore the addition if vector lengths do not match (not implemented)
        }
    }
};

#endif // ANNOY_TREE_NODE_DATA_HPP
