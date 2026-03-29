#include "PointCloud.h"
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace chrono;

// Usage: ./a2 input.txt knn
//        ./a2 input.txt approx_knn
//        ./a2 input.txt kmeans

static void printSeparator() { cout << string(44, '-') << "\n"; }

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <input_file> <knn|approx_knn|kmeans>\n";
        return 1;
    }

    try {
        string inputFile = argv[1];
        string mode      = argv[2];

        printSeparator();
        cout << "  MODE : " << mode << "\n";
        cout << "  INPUT: " << inputFile << "\n";
        printSeparator();

        PointCloud cloud;
        cloud.loadFromFile(inputFile);

        auto start = high_resolution_clock::now();

        if (mode == "knn")
            cloud.runExactKNN();
        else if (mode == "approx_knn")
            cloud.runApproxKNN();
        else if (mode == "kmeans")
            cloud.runKMeans();
        else {
            cerr << "[Error] Unknown mode: " << mode << ". Use knn, approx_knn, or kmeans.\n";
            return 1;
        }

        double ms = duration<double, milli>(high_resolution_clock::now() - start).count();

        printSeparator();
        cout << fixed << setprecision(2);
        cout << "  Points  : " << cloud.numPoints() << "\n";
        cout << "  k       : " << cloud.getK()      << "\n";
        cout << "  Runtime : " << ms                << " ms\n";
        printSeparator();

    } catch (const exception& e) {
        cerr << "[Error] " << e.what() << "\n";
        return 1;
    }

    return 0;
}