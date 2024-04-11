#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/ip.h>
#include <string>
#include <sstream>
#include <vector>
#include <random>
#include <iostream>

static void msg(const char* message) {
    fprintf(stderr, "%s\n", message);
}

static void die(const char* message) {
    int err = errno;
    fprintf(stderr, "[%d] %s\n", err, message);
    abort();
}

static int32_t read_full(int fd, char* buf, size_t n) {
    while (n > 0) {
        ssize_t rv = read(fd, buf, n);
        if (rv <= 0) return -1;
        assert((size_t)rv <= n);
        n -= (size_t)rv;
        buf += rv;
    }
    return 0;
}

std::vector<float> generateRandomVector(size_t length) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    std::vector<float> vec(length);
    for (auto& val : vec) {
        val = dis(gen);
    }
    return vec;
}

std::string generateRandomString(size_t length) {
    const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
    std::string str(length, 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, sizeof(charset) - 2);
    
    std::generate_n(str.begin(), length, [&]() { return charset[dis(gen)]; });
    return str;
}


// Serializes a vector of floats into a comma-separated string
std::string serializeVector(const std::vector<float>& vec) {
    std::stringstream ss;
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) ss << ",";
        ss << vec[i];
    }
    return ss.str();
}

static int32_t write_all(int fd, const char* buf, size_t n) {
    while (n > 0) {
        ssize_t rv = write(fd, buf, n);
        if (rv <= 0) return -1;
        assert((size_t)rv <= n);
        n -= (size_t)rv;
        buf += rv;
    }
    return 0;
}

const size_t k_max_msg = 4096;

static int32_t send_req(int fd, const std::vector<std::string>& cmd) {
    uint32_t len = 4; 
    for (const std::string& s : cmd) {
        len += 4 + s.size(); 
    }

    if (len > k_max_msg) {
        msg("Message too long");
        return -1;
    }

    char wbuf[4 + k_max_msg];
    memcpy(&wbuf[0], &len, 4);
    uint32_t n = cmd.size();
    memcpy(&wbuf[4], &n, 4);
    size_t cur = 8;
    for (const std::string& s : cmd) {
        uint32_t p = (uint32_t)s.size();
        memcpy(&wbuf[cur], &p, 4);
        memcpy(&wbuf[cur + 4], s.data(), s.size());
        cur += 4 + s.size();
    }
    return write_all(fd, wbuf, len + 4); 
}

static int32_t read_res(int fd) {
    char rbuf[4 + k_max_msg + 1]; 
    errno = 0;
    if (int32_t err = read_full(fd, rbuf, 4)) { 
        msg("Failed to read response length");
        return err;
    }

    uint32_t len = 0;
    memcpy(&len, rbuf, 4);
    if (len > k_max_msg) {
        msg("Response too long");
        return -1;
    }

    if (int32_t err = read_full(fd, &rbuf[4], len)) { 
        msg("Failed to read response body");
        return err;
    }

    uint32_t rescode = 0;
    memcpy(&rescode, &rbuf[4], 4); 
    printf("server says: [%u] %.*s\n", rescode, len - 4, &rbuf[8]); 
    return 0;
}

int main(int argc, char** argv) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) die("socket()");

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(1234); 
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK); 

    if (connect(fd, (const struct sockaddr*)&addr, sizeof(addr))) {
        die("connect in here");
    }

    // Check if the command is 'generate'
    if (argc > 1 && std::string(argv[1]) == "generate") {
        // Generate and upload 5000 random strings and vectors
        for (int i = 0; i < 1000; ++i) {
            std::vector<std::string> cmd;
            cmd.push_back("add_to_collection"); // Assuming the command to add to collection is 'add'
            cmd.push_back("collection_name"); // Assuming the collection name is specified here
            cmd.push_back(generateRandomString(10)); // Generate a random string of length 10
            cmd.push_back(serializeVector(generateRandomVector(10))); // Generate a random vector of length 5 and serialize it
            
            int32_t err = send_req(fd, cmd);
            if (err) {
                std::cerr << "Failed to send data to server" << std::endl;
                break; // Stop if there's an error
            }
            err = read_res(fd);
            if (err) {
                std::cerr << "Failed to read response from server" << std::endl;
                break; // Stop if there's an error
            }
        }
    } else {
        // Handle other commands as before
        std::vector<std::string> cmd;
        for (int i = 1; i < argc; ++i) {
            cmd.push_back(argv[i]);
        }
        int32_t err = send_req(fd, cmd);

        if (err) {
            goto L_DONE;
        }
        err = read_res(fd);
        if (err) {
            goto L_DONE;
        }
    }

L_DONE:
    close(fd);
    return 0;
}