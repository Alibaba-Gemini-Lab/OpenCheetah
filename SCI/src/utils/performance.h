#ifndef PERFORMANCE_H
#define PERFORMANCE_H

#include <ctime>
#include <string>
#include <iostream>
#include <stack>
#include <utility>
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::stack;
using std::pair;
using std::string;


inline double time_log(string tag) {
    static stack<pair<string, high_resolution_clock::time_point>> sentinel;
    
    if (sentinel.empty() || sentinel.top().first != tag) {
        auto start = high_resolution_clock::now();
        sentinel.push(make_pair(tag, start));
        return 0.0;
    }

    else {
        auto start = sentinel.top().second;
        auto end = high_resolution_clock::now();
        double duration = duration_cast<milliseconds>(end-start).count() * 1.0;
        std::cout << "[Time] " << tag << ": " 
            << duration 
            << " ms" << std::endl;
        sentinel.pop();
        return duration;
    }
}

#endif // PERFORMANCE_H
