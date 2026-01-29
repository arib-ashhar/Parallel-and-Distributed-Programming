#include "functions.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <random>

using namespace std;
using namespace std::chrono;

// Generate test data
vector<uint64_t> generateTestData(int size) {
    vector<uint64_t> orderBook;
    random_device rd;
    mt19937 gen(42); // Fixed seed for reproducibility
    uniform_int_distribution<uint32_t> stockDist(1, min(size/10, 1000));
    uniform_int_distribution<int> typeDist(0, 1);
    uniform_int_distribution<int> qtyDist(1, 100);
    uniform_int_distribution<int> valueDist(1, 100);
    
    for(int i = 0; i < size; i++) {
        uint32_t stockID = stockDist(gen);
        bool orderType = typeDist(gen);
        uint8_t orderQty = qtyDist(gen);
        uint8_t orderValue = valueDist(gen);
        
        // Encode without bit stuffing for simplicity
        uint64_t packet = 0;
        packet |= ((uint64_t)stockID) << 0;
        packet |= ((uint64_t)orderType) << 32;
        packet |= ((uint64_t)orderQty) << 33;
        packet |= ((uint64_t)orderValue) << 41;
        
        orderBook.push_back(packet);
    }
    
    return orderBook;
}

void benchmarkFunction(const string& name, function<void()> func, int threads) {
    omp_set_num_threads(threads);
    
    auto start = high_resolution_clock::now();
    func();
    auto end = high_resolution_clock::now();
    
    auto duration = duration_cast<milliseconds>(end - start).count();
    cout << name << " with " << threads << " threads: " << duration << " ms" << endl;
}

int main() {
    vector<int> sizes = {10000, 100000, 1000000};
    vector<int> thread_counts = {1, 2, 4, 8};
    
    ofstream csvFile("benchmark_results.csv");
    csvFile << "Function,Size,Threads,Time_ms,Speedup\n";
    
    for(int size : sizes) {
        cout << "\n=== Testing with " << size << " orders ===" << endl;
        auto orderBook = generateTestData(size);
        
        // Benchmark totalAmountTraded
        map<int, double> totalAmountTimes;
        for(int threads : thread_counts) {
            omp_set_num_threads(threads);
            
            auto start = high_resolution_clock::now();
            int64_t result = totalAmountTraded(orderBook);
            auto end = high_resolution_clock::now();
            
            double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
            totalAmountTimes[threads] = time_ms;
            
            double speedup = totalAmountTimes[1] / time_ms;
            csvFile << "totalAmountTraded," << size << "," << threads << "," 
                    << time_ms << "," << speedup << "\n";
            
            cout << "totalAmountTraded [" << threads << " threads]: " 
                 << time_ms << " ms (speedup: " << speedup << "x)" << endl;
        }
        
        // Benchmark printOrderStats
        map<int, double> statsTimes;
        for(int threads : thread_counts) {
            omp_set_num_threads(threads);
            
            auto start = high_resolution_clock::now();
            printOrderStats(orderBook);
            auto end = high_resolution_clock::now();
            
            double time_ms = duration_cast<milliseconds>(end - start).count();
            statsTimes[threads] = time_ms;
            
            double speedup = statsTimes[1] / time_ms;
            csvFile << "printOrderStats," << size << "," << threads << "," 
                    << time_ms << "," << speedup << "\n";
            
            cout << "printOrderStats [" << threads << " threads]: " 
                 << time_ms << " ms (speedup: " << speedup << "x)" << endl;
        }
        
        // Benchmark updateDisplay (small freq for testing)
        int freq = size / 10;
        map<int, double> displayTimes;
        for(int threads : thread_counts) {
            omp_set_num_threads(threads);
            
            auto start = high_resolution_clock::now();
            updateDisplay(orderBook, freq);
            auto end = high_resolution_clock::now();
            
            double time_ms = duration_cast<milliseconds>(end - start).count();
            displayTimes[threads] = time_ms;
            
            double speedup = displayTimes[1] / time_ms;
            csvFile << "updateDisplay," << size << "," << threads << "," 
                    << time_ms << "," << speedup << "\n";
            
            cout << "updateDisplay [" << threads << " threads]: " 
                 << time_ms << " ms (speedup: " << speedup << "x)" << endl;
        }
    }
    
    csvFile.close();
    cout << "\nResults saved to benchmark_results.csv" << endl;
    
    return 0;
}