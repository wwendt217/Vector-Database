#include "VectorSearchEngine.hpp"
#include "Algorithms/AnnoyTreeForest.hpp"
#include "Algorithms/InvertedFileIndex.hpp"
#include "Algorithms/HNSW_graph.hpp"
#include "Algorithms/Vamana.hpp"
#include <iostream>
#include <algorithm> // For std::replace in addAlgorithm
#include <chrono>    // For getCurrentTimestamp
#include <ctime>     // For std::put_time
#include <iomanip>   // For std::put_time
#include <random>

// Include the mock-up header or the actual VectorSearchAlgorithm implementations here

// Example derived class for demonstration
template<typename T>
class ExampleVectorSearchAlgorithm : public VectorSearchAlgorithm<T> {
public:
    void initialize(const std::vector<std::pair<T, std::vector<float>>>& data) override {
        std::cout << "Initializing algorithm with " << data.size() << " data points." << std::endl;
    }

    std::vector<T> search(const std::vector<float>& queryVector, int ef) const override {
        std::cout << "Performing search with ef=" << ef << std::endl;
        return {}; // Mock result
    }
};

// Function to generate a random string
std::string generateRandomString(size_t length) {
    const std::string charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(0, charset.length() - 1);
    std::string result;
    result.reserve(length);
    for (size_t i = 0; i < length; ++i) {
        result.push_back(charset[distr(gen)]);
    }
    return result;
}

// Function to generate a random vector of floats
std::vector<float> generateRandomVector(size_t length) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distr(0.0f, 1.0f);
    std::vector<float> result;
    result.reserve(length);
    for (size_t i = 0; i < length; ++i) {
        result.push_back(distr(gen));
    }
    return result;
}

int main() {
    // Create an instance of the VectorSearchEngine
    VectorSearchEngine<std::string> engine;

    // Create a collection
    std::string collectionName = "ExampleCollection";
    engine.createCollection(collectionName);

    int num_values = 5000;
    int vector_length = 10;

    std::cout << "Adding to collection..." << std::endl; 

    // Generate random node values
    std::vector<std::pair<std::string, std::vector<float>>> nodeValues;
    for (int i = 0; i < num_values; ++i) {
        std::string randomString = generateRandomString(10);
        std::vector<float> randomVector = generateRandomVector(vector_length);
        engine.addToCollection (collectionName, randomString, randomVector) ;
    }

    std::cout << "Done." << std::endl; 

    std::cout << "Creating HNSW." << std::endl; 
    float mL = 0.9f; // Example parameter, adjust as necessary
    constexpr int vector_len = 10; // Example parameter
    int num_layers = 5; // Example parameter
    int efc = 6; // Example parameter
    engine.addAlgorithm<HNSW_graph<std::string>>("h1", collectionName, mL, vector_len, num_layers, efc);
    std::cout << "Done." << std::endl; 

    std::cout << "Creating Vamana." << std::endl; 
    constexpr int num_edges = 10;
    float alpha = 0.9f;
    engine.addAlgorithm<Vamana<std::string>>("v1", collectionName, alpha, vector_length, num_edges);
    std::cout << "Done." << std::endl; 

    std::cout << "Creating IFI." << std::endl; 
    constexpr int num_centroids = 10; // Adjust according to your needs
    constexpr int retrain_threshold = 100; // Adjust according to your needs
    engine.addAlgorithm<InvertedFileIndex<std::string>>("ifi1", collectionName, vector_length, num_centroids, retrain_threshold);
    std::cout << "Done." << std::endl; 

    std::cout << "Creating ANNOY Tree Forest." << std::endl; 
    constexpr int sufficient_bucket_threshold = 200;
    constexpr int max_depth = 1000;
    constexpr int n_trees = 5;
    float threshold = 0.0f;
    engine.addAlgorithm<AnnoyTreeForest<std::string>>("annoy1", collectionName, vector_length, threshold, sufficient_bucket_threshold, max_depth, n_trees, true);
    std::cout << "Done." << std::endl; 

    std::vector<float> queryVector = generateRandomVector(vector_length);

    int ef = 10;

    // Assuming engine is an instance of VectorSearchEngine or similar
    // First, obtain a list of all algorithm names
    auto algorithmNames = engine.listAlgorithmNames();

    // Now, iterate over each algorithm name and perform a query
    for (const auto& algName : algorithmNames) {
        std::cout << "Results from " << algName << ":\n";
        auto searchResults = engine.queryAlgorithm(algName, queryVector, ef);
        
        // Assuming searchResults is a vector of pairs, with each pair being <std::string, std::vector<float>>
        for (const auto& result : searchResults) {
            std::cout << " - " << result.first << ": [";
            for (size_t i = 0; i < result.second.size(); ++i) {
                std::cout << result.second[i];
                if (i < result.second.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << std::endl; 
        }
        std::cout << std::endl;
    }

    std::cout << "Done testing VectorSearchEngine class." << std::endl; 
    engine.start_server();
    while (true) {} // busy wait...
    std::cout << "Now testing server functionality..." << std::endl; 

    std::cout << "Done." << std::endl; 

    return 0;
}
