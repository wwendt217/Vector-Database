#include <map>
#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <algorithm> // For std::find_if
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/ip.h>
#include <cstring>
#include <string>
#include <vector>
#include <sstream>
#include <map>

#include "Algorithms/HNSW_graph.hpp"
#include "Algorithms/AnnoyTreeForest.hpp"
#include "Algorithms/InvertedFileIndex.hpp"
#include "Algorithms/Vamana.hpp"
#include "Algorithms/VectorSearchAlgorithm.hpp"

#include <mutex>

template<typename T>
class VectorSearchEngine {
private:

    struct Collection {
        std::vector<std::pair<T, std::vector<float>>> data;
        std::shared_ptr<HNSW_graph<T>> hnswGraph;

        Collection(int reserveSize = 5000) 
            : data(), 
            hnswGraph(std::make_shared<HNSW_graph<T>>()) {  // Directly create HNSW_graph instance
            data.reserve(reserveSize);
        }
    };

    std::map < std::string, Collection > collections;
    std::map < std::string, std::shared_ptr<VectorSearchAlgorithm<T>> > algorithms;

 
public:

    std::mutex mtx;

    VectorSearchEngine() { }

    void createCollection(const std::string& collectionName, int reserveSize = 5000) {
        if (collections.find(collectionName) == collections.end()) {
            Collection newCollection(reserveSize);
            // Assuming HNSW_graph's constructor requires parameters
            float mL = 0.9f; // Example parameter, adjust as necessary
            int vector_len = 128; // Example parameter
            int num_layers = 5; // Example parameter
            int efc = 6; // Example parameter
            // Create and assign a new HNSW_graph instance to the collection
            newCollection.hnswGraph = std::make_shared<HNSW_graph<T>>(newCollection.data, mL, vector_len, num_layers, efc);

            collections[collectionName] = std::move(newCollection);
        
        } else {
            std::cerr << "Collection " << collectionName << " already exists.\n";
        }
    }

    // Delete a collection
    bool deleteCollection(const std::string& collectionName) {
        // Check if the collection exists
        auto it = collections.find(collectionName);
        if (it != collections.end()) {
            // The collection exists, proceed to delete
            collections.erase(it);
            return true; // Indicate successful deletion
        } else {
            // The collection does not exist, return false or optionally handle the error
            std::cerr << "Collection '" << collectionName << "' not found.\n";
            return false; // Indicate failure to delete (collection not found)
        }
    }

    // Add to a collection
    bool addToCollection(const std::string& collectionName, const T& key, const std::vector<float>& values) {
        auto it = collections.find(collectionName);
        if (it != collections.end()) {
            // Collection exists, add the data point to it
            it->second.data.emplace_back(key, values);
        
            // Optionally, update the HNSW_graph for this collection if needed
            // This would require calling a method on it->second.hnswGraph
            // For example:
            // it->second.hnswGraph->addData(key, values);
            it->second.hnswGraph->insert( std::make_pair (key, values) );
            return true; // Indicate successful addition
        } else {
            // Handle the case where the collection does not exist
            std::cerr << "Collection '" << collectionName << "' not found. Creating a new collection.\n";
        
            // Option 1: Return false to indicate failure
            // return false;
        
            // Option 2: Create the collection and add the data point
            createCollection(collectionName); // Ensure this method exists and is properly implemented
            return addToCollection(collectionName, key, values); // Retry adding after creating the collection
        }
    }

    // Delete from a collection
    bool deleteFromCollection(const std::string& collectionName, const T& key) {
        // First, find the specified collection by name.
        auto collectionIt = collections.find(collectionName);
        if (collectionIt == collections.end()) {
            std::cerr << "Collection '" << collectionName << "' not found.\n";
            return false; // Collection does not exist.
        }

        // Now, find the data point by key within the collection.
        auto& dataPoints = collectionIt->second.data; // Reference to the collection's data vector.
        auto dataPointIt = std::find_if(dataPoints.begin(), dataPoints.end(),
                                        [&key](const std::pair<T, std::vector<float>>& item) {
                                            return item.first == key;
                                        });

        if (dataPointIt == dataPoints.end()) {
            std::cerr << "Data point with key '" << key << "' not found in collection '" << collectionName << "'.\n";
            return false; // Data point does not exist within the collection.
        }

        // The data point exists; remove it from the collection.
        dataPoints.erase(dataPointIt);

        // Optionally, if the HNSW_graph needs to be updated to reflect the deletion,
        // you would call the appropriate method on the HNSW_graph instance here.
        // Example (assuming such a method exists):
        // collectionIt->second.hnswGraph->removeData(key);

        return true; // Data point successfully deleted.
    }

    std::vector<std::pair<T, std::vector<float>>> queryCollection(const std::string& collectionName, const std::vector<float>& queryVector, int ef) {
        // Check if the collection exists
        auto it = collections.find(collectionName);
        if (it == collections.end()) {
            std::cerr << "Collection '" << collectionName << "' not found.\n";
            return {}; // Return an empty vector to indicate failure
        }

        // Check if the HNSW_graph is initialized
        if (!it->second.hnswGraph) {
            std::cerr << "HNSW_graph for collection '" << collectionName << "' is not initialized.\n";
            return {}; // Return an empty vector to indicate failure
        }

        // Perform the query on the HNSW_graph associated with the collection
        auto& hnswGraph = it->second.hnswGraph;
        // Ensure the HNSW_graph has a method `searchClosest` that matches the expected signature
        try {
            return hnswGraph->searchClosest(queryVector, ef);
        } catch (const std::exception& e) {
            // Catch exceptions if searchClosest could throw
            std::cerr << "An error occurred during the query: " << e.what() << '\n';
            return {}; // Return an empty vector to indicate failure
        }
    }

    template<typename Alg, typename... Args>
    std::string addAlgorithm(const std::string& algName, const std::string& name, Args&&... args) {
        // Check if the collection exists
        auto it = collections.find(name);
        if (it == collections.end()) {
            std::cerr << "Collection '" << name << "' not found.\n";
            return {}; // Return an empty vector to indicate failure
        }

        // Ensure T is derived from VectorSearchEngine
        static_assert(std::is_base_of<VectorSearchAlgorithm<T>, Alg>::value, "T must inherit from VectorSearchEngine");

        // Create a new instance of T, passing in the forwarded arguments
        auto algorithm = std::make_shared<Alg>(it->second.data, std::forward<Args>(args)...);

        // Generate a unique name for the algorithm
        std::string uniqueName = name;
        int counter = 1;
        while (algorithms.find(uniqueName) != algorithms.end()) {
            uniqueName = name + "_" + std::to_string(counter);
            ++counter;
        }

        // Add the newly created algorithm instance to the map
        algorithms.emplace(algName, std::move(algorithm));

        // Return the name for confirmation or further use
        return uniqueName;
    }

    // Method to list all algorithm names
    std::vector<std::string> listAlgorithmNames() const {
        std::vector<std::string> names;
        for (const auto& algPair : algorithms) {
            names.push_back(algPair.first);
        }
        return names;
    }

    // Method to list all algorithm names
    std::vector<std::string> listCollectionNames() const {
        std::vector<std::string> names;
        for (const auto& algPair : collections) {
            names.push_back(algPair.first);
        }
        return names;
    }

    // Method that takes an algorithm name, a query vector, and ef, then calls searchClosest
    std::vector<std::pair<std::string, std::vector<float>>> queryAlgorithm(const std::string& algName, const std::vector<float>& queryVector, int ef) {
        auto it = algorithms.find(algName);
        if (it != algorithms.end()) {
            // Algorithm found, perform the query
            return it->second->searchClosest(queryVector, ef);
        } else {
            // Algorithm not found, handle the error or return an empty result
            std::cerr << "Algorithm '" << algName << "' not found.\n";
            return {};
        }
    }

    /* Server Functionality */

    static void msg (const char* msg) { fprintf(stderr, "%s\n", msg); }

    static void die (const char* msg) {
        fprintf (stderr, "[%d] %s\n", (int) errno, msg);
        abort ();
    }

    static const size_t k_max_msg = 4096;
    static const size_t k_max_args = 1024;

    enum {
        STATE_REQ = 0,
        STATE_RES = 1,
        STATE_END = 2
    };

    enum {
        RES_OK = 0,
        RES_ERR = 1,
        RES_NX = 2    
    };

    struct Conn {
        int fd = -1;
        uint32_t state = 0;
        size_t rbuf_size = 0;
        uint8_t rbuf[4 + k_max_msg];
        size_t wbuf_size = 0;
        size_t wbuf_sent = 0;
        uint8_t wbuf[4 + k_max_msg];
    };

    static int32_t conn_put (std::vector<Conn *> &fd2conn, struct Conn* conn) {
        if (fd2conn.size() <= (size_t) conn->fd) {
            fd2conn.resize (conn->fd + 1);
        }
        fd2conn[conn->fd] = conn;
    }

    static void fd_set_nb (int fd) {
        errno = 0;
        int flags = fcntl (fd, F_GETFL, 0);
        if (errno) {
            die ("fcntl error");
            return;
        }

        flags |= O_NONBLOCK;

        errno = 0;
        fcntl (fd, F_SETFL, flags);
        if (errno) { die ("fcntl error in fd_set_nb."); }
    }

    static bool try_flush_buffer (Conn* conn) {
        ssize_t rv = 0;
        do {
            size_t remain = conn->wbuf_size - conn->wbuf_sent;
            rv = write(conn->fd, &conn->wbuf[conn->wbuf_sent], remain);
        } while (rv < 0 && errno == EINTR);
        if (rv < 0 && errno == EAGAIN) {
            return false;
        }
        if (rv < 0) {
            msg("write() error");
            conn->state = STATE_END;
            return false;
        }
        conn->wbuf_sent += (size_t) rv;
        assert (conn->wbuf_sent <= conn->wbuf_size);
        if (conn->wbuf_sent == conn->wbuf_size) {
            conn->state = STATE_REQ;
            conn->wbuf_sent = 0;
            conn->wbuf_size = 0;
            return false;
        }
        return true;
    }

    static void state_res (Conn* conn) {
        while (try_flush_buffer (conn)) {}
    }

    static bool cmd_is(const std::string &word, const char *cmd) {
        return 0 == strcasecmp(word.c_str(), cmd);
    }

    uint32_t create_collection (
        const std::vector<std::string>& cmd, uint8_t* res, uint32_t* reslen
    ) {
        // Check if the key already exists in the map
        if (collections.find(cmd[1]) == collections.end()) {
            // Key does not exist, so add it with a new empty vector
            createCollection(cmd[1]);
            std::cout << "Added new entry with key: " << cmd[1] << std::endl;
            return RES_OK;
        } else {
            std::cout << "Key already exists: " << cmd[1] << std::endl;
        }    
    }

    uint32_t add_to_collection(
        const std::vector<std::string>& cmd, uint8_t* res, uint32_t* reslen
    ) {
        // Check if the cmd vector has the expected number of arguments
        if (cmd.size() < 4) {
            std::cout << "Insufficient arguments provided." << std::endl;
            return 2; // Error code for insufficient arguments
        }
    
        // Check if the specified collection exists
        if (collections.find(cmd[1]) == collections.end()) {
            std::cout << "Collection does not exist: " << cmd[1] << std::endl;
            return 1; // Error code for non-existing collection
        }

        // Parse the string of floats
        std::vector<float> floats;
        std::stringstream ss(cmd[3]);
        std::string item;
        while (std::getline(ss, item, ',')) {
            try {
                floats.push_back(std::stof(item));
            } catch (...) {
                std::cout << "Invalid float in list: " << item << std::endl;
                return 3; // Error code for invalid float
            }
        }

        // Add the new string and vector of floats as a pair to the specified collection
        addToCollection(cmd[1], cmd[2], floats);
        std::cout << "Added to collection: " << cmd[1] << std::endl;

        // Success
        return RES_OK;
    }

    uint32_t query_collection(
        const std::vector<std::string>& cmd, uint8_t* res, uint32_t* reslen
    ) {
        if (cmd.size() < 3) {
            std::cerr << "Insufficient arguments" << std::endl;
            return 1; // Error code for insufficient arguments
        }

        // Parse the string of floats from cmd[2]
        std::vector<float> queryVec;
        std::stringstream ss(cmd[2]);
        std::string item;
        while (std::getline(ss, item, ',')) {
            try {
                queryVec.push_back(std::stof(item));
            } catch (...) {
                std::cerr << "Invalid float in list: " << item << std::endl;
                return 3; // Error code for invalid float
            }
        }

        // Perform the search
        auto searchResults = queryCollection(cmd[1], queryVec, std::stoi(cmd[3])); // Assuming we want the top 10 results

        // Here, handle the searchResults according to your application's requirements
        // This example just prepares a string value from the search results to send back
        //std::string val = searchResults[0]->value.first; // Replace this with actual processing logic to extract value from searchResults
        // Assuming searchResults is something like std::vector<YourResultType*>
        std::string val;
        for (auto& result : searchResults) {
            if (!val.empty()) {
                val += "\n"; // Add a newline between strings, but not before the first string
            }
            val += result.first; // Append the current string
        }
        assert(val.size() <= k_max_msg);
        memcpy(res, val.data(), val.size());
        *reslen = (uint32_t) val.size();
 
        return RES_OK; // Success
    }

    uint32_t queryAlg(
        const std::vector<std::string>& cmd, uint8_t* res, uint32_t* reslen
    ) {
        if (cmd.size() < 3) {
            std::cerr << "Insufficient arguments" << std::endl;
            return 1; // Error code for insufficient arguments
        }

        // Parse the string of floats from cmd[2]
        std::vector<float> queryVec;
        std::stringstream ss(cmd[2]);
        std::string item;
        while (std::getline(ss, item, ',')) {
            try {
                queryVec.push_back(std::stof(item));
            } catch (...) {
                std::cerr << "Invalid float in list: " << item << std::endl;
                return 3; // Error code for invalid float
            }
        }

        // Perform the search
        auto searchResults = queryAlgorithm(cmd[1], queryVec, std::stoi(cmd[3])); // Assuming we want the top 10 results

        // Here, handle the searchResults according to your application's requirements
        // This example just prepares a string value from the search results to send back
        //std::string val = searchResults[0]->value.first; // Replace this with actual processing logic to extract value from searchResults
        // Assuming searchResults is something like std::vector<YourResultType*>
        std::string val;
        for (auto& result : searchResults) {
            if (!val.empty()) {
                val += "\n"; // Add a newline between strings, but not before the first string
            }
            val += result.first; // Append the current string
        }
        assert(val.size() <= k_max_msg);
        memcpy(res, val.data(), val.size());
        *reslen = (uint32_t) val.size();
 
        return RES_OK; // Success
    }

    uint32_t listAlgorithms (
        const std::vector<std::string>& cmd, uint8_t* res, uint32_t* reslen
    ) {
        auto searchResults = listAlgorithmNames();
        std::string val;
        for (auto& result : searchResults) {
            if (!val.empty()) {
                val += "\n"; // Add a newline between strings, but not before the first string
            }
            val += result; // Append the current string
        }
        assert(val.size() <= k_max_msg);
        memcpy(res, val.data(), val.size());
        *reslen = (uint32_t) val.size();
 
        return RES_OK; // Success
    }

    uint32_t listCollections (
        const std::vector<std::string>& cmd, uint8_t* res, uint32_t* reslen
    ) {
        auto searchResults = listCollectionNames();
        std::string val;
        for (auto& result : searchResults) {
            if (!val.empty()) {
                val += "\n"; // Add a newline between strings, but not before the first string
            }
            val += result; // Append the current string
        }
        assert(val.size() <= k_max_msg);
        memcpy(res, val.data(), val.size());
        *reslen = (uint32_t) val.size();
 
        return RES_OK; // Success
    }

    uint32_t addHNSW (
        const std::vector<std::string>& cmd, uint8_t* res, uint32_t* reslen
    ) {
        if (cmd.size() < 6) {
            std::cerr << "Insufficient arguments" << std::endl;
            return 1; // Error code for insufficient arguments
        }

        std::string collectionName = cmd[1];
        std::string algName = cmd[2];
        float mL = std::stof(cmd[3]);
        int vector_len = std::stoi(cmd[4]); 
        int num_layers = std::stoi(cmd[5]); 
        int efc = std::stoi(cmd[6]); 

        std::cout << "Building HNSW for " << collectionName << std::endl;

        addAlgorithm<HNSW_graph<std::string>>(algName, collectionName, mL, vector_len, num_layers, efc);

        std::cout << "HNSW graph built for collection: " << collectionName << std::endl;

        return RES_OK; // Success
    }

    uint32_t addANNOY (
        const std::vector<std::string>& cmd, uint8_t* res, uint32_t* reslen
    ) {
        if (cmd.size() < 8) {
            std::cerr << "Insufficient arguments" << std::endl;
            return 1; // Error code for insufficient arguments
        }

        std::string collectionName = cmd[1];
        std::string algName = cmd[2];
        int vector_len = std::stoi(cmd[3]); 
        float sufficient_bucket_threshold = std::stoi(cmd[5]);
        int max_depth = std::stoi(cmd[6]); 
        int n_trees = std::stoi(cmd[7]); 
        float threshold = std::stof(cmd[4]);

        std::cout << "Building ANNOY for " << collectionName << std::endl;

        addAlgorithm<AnnoyTreeForest<std::string>>(algName, collectionName, vector_len, threshold, sufficient_bucket_threshold, max_depth, n_trees, true);

        std::cout << "ANNOY built for collection: " << collectionName << std::endl;

        return RES_OK; // Success
    }

    uint32_t addIFI (
        const std::vector<std::string>& cmd, uint8_t* res, uint32_t* reslen
    ) {
        if (cmd.size() < 5) {
            std::cerr << "Insufficient arguments" << std::endl;
            return 1; // Error code for insufficient arguments
        }

        std::string collectionName = cmd[1];
        std::string algName = cmd[2];
        int vector_length = std::stoi(cmd[3]);
        int num_centroids = std::stoi(cmd[4]); // Adjust according to your needs
        int retrain_threshold = std::stoi(cmd[5]); // Adjust according to your needs

        std::cout << "Building InvertedFileIndex for " << collectionName << std::endl;

        addAlgorithm<InvertedFileIndex<std::string>>(algName, collectionName, vector_length, num_centroids, retrain_threshold);

        std::cout << "InvertedFileIndex built for collection: " << collectionName << std::endl;

        return RES_OK; // Success
    }

    uint32_t addVamana (
        const std::vector<std::string>& cmd, uint8_t* res, uint32_t* reslen
    ) {
        if (cmd.size() < 5) {
            std::cerr << "Insufficient arguments" << std::endl;
            return 1; // Error code for insufficient arguments
        }
        
        std::string collectionName = cmd[1];
        std::string algName = cmd[2];
        int vector_length = std::stoi(cmd[3]);
        int num_edges = std::stoi(cmd[4]);
        float alpha = std::stof(cmd[5]);
        
        std::cout << "Building Vamana for " << collectionName << std::endl;

        addAlgorithm<Vamana<std::string>>(algName, collectionName, alpha, vector_length, num_edges);

        std::cout << "Vamana built for collection: " << collectionName << std::endl;

        return RES_OK; // Success
    }

    static int32_t parse_req(
        const uint8_t* data, size_t len, std::vector<std::string>& out)
    {
        if (len < 4) {
            return -1;
        }
        uint32_t n = 0;
        memcpy (&n, &data[0], 4);
        if (n > k_max_args) {
            return -1;
        }

        size_t pos = 4;
        while (n--) {
            if (pos + 4 > len) { return -1; }
            uint32_t sz = 0;
            memcpy(&sz, &data[pos], 4);
            if (pos + 4 + sz > len) { return -1; }
            out.push_back(std::string((char*) &data[pos+4], sz));
            pos += 4 + sz;
        }

        if (pos != len) { return -1; }
    
        return 0;
    }

    int32_t do_request (
        const uint8_t *req, uint32_t reqlen, 
        uint32_t *rescode, uint8_t *res, uint32_t *reslen)
    {
        std::vector<std::string> cmd;
        if (0 != parse_req(req, reqlen, cmd)) {
            msg("Bad req");
            return -1;
        }

        // Handling "query" command for querying a collection
        if (cmd.size() >= 3 && cmd_is(cmd[0], "query")) {
            *rescode = query_collection(cmd, res, reslen);
        }    
        // Handling "create_collection" command for creating a new collection
        else if (cmd.size() == 2 && cmd_is(cmd[0], "create_collection")) {
            *rescode = create_collection(cmd, res, reslen);
        }
        // Handling "add_to_collection" command for adding to an existing collection
        else if (cmd.size() >= 4 && cmd_is(cmd[0], "add_to_collection")) {
            *rescode = add_to_collection(cmd, res, reslen);
        }
        // Commands for buildings algorithms from collections
        else if (cmd.size() >= 4 && cmd_is(cmd[0], "Vamana")) {
            *rescode = addVamana(cmd, res, reslen);
        }
        else if (cmd.size() >= 4 && cmd_is(cmd[0], "HNSW")) {
            *rescode = addHNSW(cmd, res, reslen);
        }
        else if (cmd.size() >= 4 && cmd_is(cmd[0], "IFI")) {
            *rescode = addIFI(cmd, res, reslen);
        }
        else if (cmd.size() >= 4 && cmd_is(cmd[0], "ANNOY")) {
            *rescode = addANNOY(cmd, res, reslen);
        }
        else if (cmd.size() >= 4 && cmd_is(cmd[0], "queryAlg")) {
            *rescode = queryAlg(cmd, res, reslen);
        }
        else if (cmd_is(cmd[0], "Collections")) {
            *rescode = listCollections(cmd, res, reslen);
        }
        else if (cmd_is(cmd[0], "Algorithms")) {
            *rescode = listAlgorithms(cmd, res, reslen);
        }
        else if (cmd_is(cmd[0], "exit")) {
            std::exit(0);
            return 0;
        }
        else {
            *rescode = RES_ERR;
            strcpy((char*) res, "Unknown cmd.");
            *reslen = strlen("Unknown cmd."); // Fixed typo in the original code.
            return 0;
        }
        return 0;
    }

    static int32_t accept_new_conn (std::vector<Conn*> &fd2conn, int fd) {
        struct sockaddr_in client_addr = {};
        socklen_t socklen = sizeof(client_addr);
        int connfd = accept (fd, (struct sockaddr *) &client_addr, &socklen);
        if (connfd < 0) {
            msg ("Accept error.");
            return -1;
        }

        fd_set_nb (connfd);

        struct Conn *conn = (struct Conn *) malloc(sizeof(struct Conn));
        if (!conn) {
            close (connfd);
            return -1;
        }
        conn->fd = connfd;
        conn->state = STATE_REQ;
        conn->rbuf_size = 0;
        conn->wbuf_size = 0;
        conn->wbuf_sent = 0;
        conn_put (fd2conn, conn);
        return 0;
    }

    bool try_one_request (Conn* conn) {
        if (conn->rbuf_size < 4) { return false; }
        uint32_t len = 0;
        memcpy (&len, &conn->rbuf[0], 4);
        if (len > k_max_msg) {
            msg("too long");
            conn->state = STATE_END;
            return false;
        }
        if (4 + len > conn->rbuf_size) {
            return false;
        }
        uint32_t rescode = 0;
        uint32_t wlen = 0;
        int32_t err = do_request (
            &conn->rbuf[4], len,
            &rescode, &conn->wbuf[4 + 4], &wlen
        );
        if (err) {
            conn->state = STATE_END;
            return false;
        }
        wlen += 4;
        memcpy (&conn->wbuf[0], &wlen, 4);
        memcpy (&conn->wbuf[4], &rescode, 4);
        conn->wbuf_size = 4 + wlen;

        size_t remain = conn->rbuf_size - 4 - len;
        if (remain) {
            memmove (conn->rbuf, &conn->rbuf[4 + len], remain);
        }
        conn->rbuf_size = remain;

        conn->state = STATE_RES;
        state_res (conn);

        return (conn->state == STATE_REQ);
    }

    bool try_fill_buffer (Conn *conn) {
        assert (conn->rbuf_size < sizeof(conn->rbuf));
        ssize_t rv = 0;
        do {
            size_t cap = sizeof (conn->rbuf) - conn->rbuf_size;
            rv = read(conn->fd, &conn->rbuf[conn->rbuf_size], cap);
        } while (rv < 0 && errno == EINTR);
        if (rv < 0 && errno == EAGAIN) {
            return false;
        }
        if (rv < 0) {
            msg ("read() error");
            conn->state = STATE_END;
            return false;
        }
        if (rv == 0) {
            if (conn->rbuf_size > 0) {
                msg ("unexpected EOF");
            } else {
                msg ("EOF");
            }
            conn->state = STATE_END;
            return false;
        }

        conn->rbuf_size += (size_t) rv;
        assert (conn->rbuf_size <= sizeof(conn->rbuf));

        while (try_one_request(conn)) {}
        return (conn->state == STATE_REQ);
    }

    void state_req (Conn *conn) {
        while (try_fill_buffer (conn) ) {}
    }

    void connection_io (Conn* conn) {
        if ( conn->state == STATE_REQ ) {
            state_req (conn);
        } else if ( conn->state == STATE_RES) {
            state_res (conn);
        } else {
            assert (0); // Error encountered
        }
    }

    void serve_forever(int port_id = 1234) {
        // Socket creation and configuration
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if(fd < 0) { die("Socket couldn't be created."); }
        int val = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &val, sizeof(val));

        // Address configuration
        struct sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_port = ntohs(port_id);
        addr.sin_addr.s_addr = ntohl(0);
        int rv = bind ( fd, (const sockaddr *)& addr, sizeof(addr) );
        if (rv) { die ("Could not bind addr to port."); }

        // Listen
        rv = listen (fd, SOMAXCONN);
        if (rv) { die ("Could not listen."); }

        // Vector of client connections
        std::vector<Conn *> fd2conn;

        // Nonblocking
        fd_set_nb (fd);

        std::vector<struct pollfd> poll_args;
        while (true) {
            mtx.lock();
            poll_args.clear();
            struct pollfd pfd = {fd, POLLIN, 0};
            poll_args.push_back (pfd);

            for (Conn* conn : fd2conn) {
                if (!conn) { continue; }
            
                struct pollfd pfd = {};
                pfd.fd = conn->fd;
                pfd.events = (conn->state == STATE_REQ) ? POLLIN : POLLOUT;
                pfd.events = pfd.events | POLLERR;
                poll_args.push_back (pfd);
            }

            int rv = poll (poll_args.data(), (nfds_t) poll_args.size(), 1000);
            if (rv < 0) { die ("Poll failed in while loop."); }

            for ( size_t i = 1; i < poll_args.size(); ++i ) { 
                if ( poll_args[i].revents ) {
                    Conn* conn = fd2conn[poll_args[i].fd];
                    connection_io (conn);
                    if (conn -> state == STATE_END) {
                        fd2conn[conn->fd] = NULL;
                        close (conn->fd);
                        free (conn);
                    }
                }
            }

            if (poll_args[0].revents) {accept_new_conn (fd2conn, fd); }
            mtx.unlock();
        }
    }

    pid_t pid;
    int start_server() {
        pid = fork();
        if (pid == -1) {
            // If fork() returns -1, an error occurred
            std::cerr << "Failed to fork()" << std::endl;
            return 1;
        } else if (pid > 0) {
            // If fork() returns a positive number, we are in the parent process
            // and the return value is the PID of the newly created child process.
            std::cout << "Parent process, PID = " << getpid() << std::endl;
            std::cout << "Created a child process, PID = " << pid << std::endl;
        } else {
            // If fork() returns 0, we are in the child process
            std::cout << "Child process, PID = " << getpid() << std::endl;
            serve_forever();
            // Child process can execute different code here
            // For example, you could use exec* functions to run a different program
        }

        // Both processes continue executing here
        return 0;
    }
};
