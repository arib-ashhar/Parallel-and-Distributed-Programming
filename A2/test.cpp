#include "PointCloud.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <iomanip>
#include <chrono>

using namespace std;
using namespace chrono;

// ---- config ----------------------------------------------------------------
static const int    NUM_POINTS    = 100000;
static const int    K_NEIGHBORS   = 50;
static const int    MAX_ITER      = 20;
static const float  COORD_RANGE   = 100.0f;
static const double MAE_THRESHOLD = 3.0;

static const string INPUT_FILE         = "test_input.txt";
static const string KNN_SEQ_FILE       = "knn_sequential.txt";
static const string KNN_PAR_FILE       = "knn.txt";
static const string APPROX_KNN_FILE    = "approx_knn.txt";
static const string KMEANS_FILE        = "kmeans.txt";
static const string KMEANS_SEQ_FILE    = "kmeans_sequential.txt";

// ---- helpers ---------------------------------------------------------------
void generateInputFile(int n, int k, int maxIter) {
    srand(static_cast<unsigned>(time(nullptr)));
    ofstream file(INPUT_FILE);
    if (!file.is_open()) { cerr << "[Test] Failed to create input file.\n"; exit(1); }
    file << n << "\n" << k << "\n" << maxIter << "\n";
    for (int i = 0; i < n; ++i) {
        float x = static_cast<float>(rand()) / RAND_MAX * COORD_RANGE;
        float y = static_cast<float>(rand()) / RAND_MAX * COORD_RANGE;
        float z = static_cast<float>(rand()) / RAND_MAX * COORD_RANGE;
        file << fixed << setprecision(4) << x << " " << y << " " << z << " " << (rand() % 256) << "\n";
    }
    cout << "[Test] Generated " << n << " points → " << INPUT_FILE << "\n";
}

vector<int> loadIntensities(const string& filePath) {
    ifstream file(filePath);
    if (!file.is_open()) { cerr << "[Test] Cannot open: " << filePath << "\n"; exit(1); }
    vector<int> out;
    float x, y, z; int intensity;
    while (file >> x >> y >> z >> intensity)
        out.push_back(intensity);
    return out;
}

struct Metrics { double mae; int maxError; int mismatchCount; double ms; double speedup; };

Metrics computeMetrics(const vector<int>& ref, const vector<int>& test, double testMs, double speedupBaseMs) {
    Metrics m = {0.0, 0, 0, testMs, speedupBaseMs / testMs};
    int n = static_cast<int>(ref.size());
    for (int i = 0; i < n; ++i) {
        int diff = abs(ref[i] - test[i]);
        m.mae += diff;
        if (diff > 0) ++m.mismatchCount;
        if (diff > m.maxError) m.maxError = diff;
    }
    m.mae /= n;
    return m;
}

void printSeparator() { cout << string(52, '-') << "\n"; }

void printSection(const string& title, const Metrics& m, bool exactMatch) {
    printSeparator();
    cout << "  " << title << "\n";
    printSeparator();
    cout << fixed << setprecision(4);
    cout << "  Time               : " << m.ms           << " ms\n";
    cout << "  Speedup            : " << m.speedup       << "x\n";
    cout << "  MAE                : " << m.mae           << "\n";
    cout << "  Max error          : " << m.maxError      << "\n";
    cout << "  Mismatched points  : " << m.mismatchCount << " / " << NUM_POINTS << "\n";
    if (exactMatch) {
        if (m.mae == 0.0) cout << "  STATUS: PASSED (exact match)\n";
        else              cout << "  STATUS: FAILED (MAE=" << m.mae << ", expected 0)\n";
    } else {
        if (m.mae <= MAE_THRESHOLD) cout << "  STATUS: PASSED (MAE " << m.mae << " <= " << MAE_THRESHOLD << ")\n";
        else                        cout << "  STATUS: FAILED (MAE " << m.mae << " >  " << MAE_THRESHOLD << ")\n";
    }
}

// ---- main ------------------------------------------------------------------
int main() {
    printSeparator();
    cout << "  POINT CLOUD TEST SUITE\n";
    cout << "  N=" << NUM_POINTS << "  k=" << K_NEIGHBORS << "  maxIter=" << MAX_ITER << "\n";
    printSeparator();

    generateInputFile(NUM_POINTS, K_NEIGHBORS, MAX_ITER);
    string inputFile = INPUT_FILE;

    // 1. Sequential KNN — ground truth for exact & approx KNN
    PointCloud seqKnnCloud;
    seqKnnCloud.loadFromFile(inputFile);
    cout << "[Test] Running sequential KNN (ground truth)...\n";
    auto t0 = high_resolution_clock::now();
    seqKnnCloud.runExactKNNSequential();
    double seqKnnMs = duration<double, milli>(high_resolution_clock::now() - t0).count();
    cout << "[Test] Done: " << fixed << setprecision(1) << seqKnnMs << " ms\n\n";
    vector<int> seqKnnIntensities = loadIntensities(KNN_SEQ_FILE);

    // 2. Exact KNN — CUDA (speedup vs sequential KNN)
    PointCloud knnCloud;
    knnCloud.loadFromFile(inputFile);
    cout << "[Test] Running exact KNN (CUDA)...\n";
    t0 = high_resolution_clock::now();
    knnCloud.runExactKNN();
    double knnMs = duration<double, milli>(high_resolution_clock::now() - t0).count();
    cout << "[Test] Done: " << fixed << setprecision(1) << knnMs << " ms\n\n";
    vector<int> knnIntensities = loadIntensities(KNN_PAR_FILE);

    // 3. Approx KNN — voxel hashing (speedup vs exact KNN per spec)
    PointCloud approxCloud;
    approxCloud.loadFromFile(inputFile);
    cout << "[Test] Running approx KNN (voxel hashing)...\n";
    t0 = high_resolution_clock::now();
    approxCloud.runApproxKNN();
    double approxMs = duration<double, milli>(high_resolution_clock::now() - t0).count();
    cout << "[Test] Done: " << fixed << setprecision(1) << approxMs << " ms\n\n";
    vector<int> approxIntensities = loadIntensities(APPROX_KNN_FILE);

    // 4. Sequential K-Means — ground truth for K-Means CUDA
    PointCloud seqKmeansCloud;
    seqKmeansCloud.loadFromFile(inputFile);
    cout << "[Test] Running sequential K-Means (ground truth)...\n";
    t0 = high_resolution_clock::now();
    seqKmeansCloud.runKMeansSequential();
    double seqKmeansMs = duration<double, milli>(high_resolution_clock::now() - t0).count();
    cout << "[Test] Done: " << fixed << setprecision(1) << seqKmeansMs << " ms\n\n";
    vector<int> seqKmeansIntensities = loadIntensities(KMEANS_SEQ_FILE);

    // 5. K-Means — CUDA (speedup vs sequential K-Means)
    PointCloud kmeansCloud;
    kmeansCloud.loadFromFile(inputFile);
    cout << "[Test] Running K-Means (CUDA)...\n";
    t0 = high_resolution_clock::now();
    kmeansCloud.runKMeans();
    double kmeansMs = duration<double, milli>(high_resolution_clock::now() - t0).count();
    cout << "[Test] Done: " << fixed << setprecision(1) << kmeansMs << " ms\n\n";
    vector<int> kmeansIntensities = loadIntensities(KMEANS_FILE);

    // Validate sizes
    if ((int)seqKnnIntensities.size()    != NUM_POINTS ||
        (int)knnIntensities.size()       != NUM_POINTS ||
        (int)approxIntensities.size()    != NUM_POINTS ||
        (int)seqKmeansIntensities.size() != NUM_POINTS ||
        (int)kmeansIntensities.size()    != NUM_POINTS) {
        cerr << "[Test] Output size mismatch.\n"; return 1;
    }

    // Compute metrics
    // Exact KNN  : vs sequential KNN,    speedup vs seq KNN
    // Approx KNN : vs sequential KNN,    speedup vs exact KNN  (per spec)
    // K-Means    : vs sequential KMeans, speedup vs seq KMeans
    Metrics knnMetrics    = computeMetrics(seqKnnIntensities,    knnIntensities,    knnMs,    seqKnnMs);
    Metrics approxMetrics = computeMetrics(seqKnnIntensities,    approxIntensities, approxMs, knnMs);
    Metrics kmeansMetrics = computeMetrics(seqKmeansIntensities, kmeansIntensities, kmeansMs, seqKmeansMs);

    // Print full report
    printSeparator();
    cout << "  SUMMARY\n";
    cout << "  Seq KNN baseline   : " << fixed << setprecision(1) << seqKnnMs    << " ms\n";
    cout << "  Seq KMeans baseline: " << fixed << setprecision(1) << seqKmeansMs << " ms\n";
    printSection("EXACT KNN  (CUDA, speedup vs seq KNN)",              knnMetrics,    true);
    printSection("APPROX KNN (voxel hashing, speedup vs exact KNN)",   approxMetrics, false);
    printSection("K-MEANS    (CUDA, speedup vs seq K-Means)",          kmeansMetrics, false);
    printSeparator();

    bool allPassed = (knnMetrics.mae    == 0.0)          &&
                     (approxMetrics.mae <= MAE_THRESHOLD) &&
                     (kmeansMetrics.mae <= MAE_THRESHOLD);

    cout << "  OVERALL: " << (allPassed ? "PASSED" : "FAILED") << "\n";
    printSeparator();
    return allPassed ? 0 : 1;
}