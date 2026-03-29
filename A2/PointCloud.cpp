#include "PointCloud.h"
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace std;

PointCloud::PointCloud() : k(0), maxIter(0) {}

void PointCloud::loadFromFile(string& filePath) {
    ifstream file(filePath);
    if (!file.is_open())
        throw runtime_error("Cannot open input file: " + filePath);

    int n = 0;
    file >> n >> k >> maxIter;
    if (n <= 0 || k <= 0)
        throw runtime_error("Invalid n or k in input file.");

    points.clear();
    points.reserve(n);

    for (int i = 0; i < n; ++i) {
        float x, y, z;
        int intensity;
        if (!(file >> x >> y >> z >> intensity))
            throw runtime_error("Unexpected end of input at point " + to_string(i));
        points.emplace_back(x, y, z, intensity);
    }

    cout << "[PointCloud] Loaded " << n << " points. k=" << k << " maxIter=" << maxIter << "\n";
}

float PointCloud::squaredDist(int i, int j) const {
    float dx = points[i].x - points[j].x;
    float dy = points[i].y - points[j].y;
    float dz = points[i].z - points[j].z;
    return dx*dx + dy*dy + dz*dz;
}

vector<int> PointCloud::buildHistogram(const vector<int>& neighbors) const {
    vector<int> histogram(256, 0);
    for (int idx : neighbors)
        ++histogram[points[idx].intensity];
    return histogram;
}

int PointCloud::remapIntensity(const vector<int>& histogram, int origIntensity, int m) const {
    vector<int> cdf(256, 0);
    cdf[0] = histogram[0];
    for (int v = 1; v < 256; ++v)
        cdf[v] = cdf[v-1] + histogram[v];

    int cdfMin = 0;
    for (int v = 0; v < 256; ++v) {
        if (cdf[v] > 0) { cdfMin = cdf[v]; break; }
    }

    if (m == cdfMin)
        return origIntensity;

    float remapped = static_cast<float>(cdf[origIntensity] - cdfMin) / static_cast<float>(m - cdfMin) * 255.0f;
    return max(0, min(255, static_cast<int>(round(remapped))));
}

void PointCloud::writeOutput(const string& filePath) const {
    ofstream outFile(filePath);
    if (!outFile.is_open())
        throw runtime_error("Cannot open output file: " + filePath);

    outFile << fixed;
    for (const Point& p : points)
        outFile << p.x << " " << p.y << " " << p.z << " " << p.newIntensity << "\n";

    cout << "[PointCloud] Output written to: " << filePath << "\n";
}

// ----- shared KNN core ------------------------------------------------------
// Returns the k nearest neighbour indices for point i.
template<typename Fn>
static vector<int> findKNN(int i, int k, int n, Fn squaredDistFn) {
    using DistIdx = pair<float, int>;
    vector<DistIdx> heap;
    heap.reserve(k + 1);

    for (int j = 0; j < n; ++j) {
        if (j == i) continue;
        float d = squaredDistFn(i, j);

        if (static_cast<int>(heap.size()) < k) {
            heap.push_back({d, j});
            if (static_cast<int>(heap.size()) == k)
                make_heap(heap.begin(), heap.end());
        } else if (d < heap.front().first || (d == heap.front().first && j < heap.front().second)) {
            pop_heap(heap.begin(), heap.end());
            heap.back() = {d, j};
            push_heap(heap.begin(), heap.end());
        }
    }

    vector<int> neighbors;
    neighbors.reserve(heap.size());
    for (const auto& [dist, idx] : heap)
        neighbors.push_back(idx);
    return neighbors;
}

// ----- sequential KNN (ground truth) ----------------------------------------
void PointCloud::runExactKNNSequential() {
    int n = numPoints();

    for (int i = 0; i < n; ++i) {
        auto neighbors = findKNN(i, k, n, [&](int a, int b){ return squaredDist(a, b); });
        neighbors.push_back(i);
        auto histogram = buildHistogram(neighbors);
        points[i].newIntensity = remapIntensity(histogram, points[i].intensity, k + 1);
    }

    writeOutput("knn_sequential.txt");
}


// ----- sequential K-Means (ground truth) -------------------------------------
void PointCloud::runKMeansSequential() {
    int n = numPoints();

    // Init centroids = first k points
    vector<float> cX(k), cY(k), cZ(k);
    for (int c = 0; c < k; ++c) {
        cX[c] = points[c].x; cY[c] = points[c].y; cZ[c] = points[c].z;
    }

    vector<int> clusterIds(n, 0);

    // Iterate
    for (int iter = 0; iter < maxIter; ++iter) {
        bool changed = false;

        // Assignment step
        for (int i = 0; i < n; ++i) {
            float bestDist = 1e30f;
            int   bestId   = 0;
            for (int c = 0; c < k; ++c) {
                float dx = points[i].x - cX[c], dy = points[i].y - cY[c], dz = points[i].z - cZ[c];
                float d  = dx*dx + dy*dy + dz*dz;
                if (d < bestDist) { bestDist = d; bestId = c; }
            }
            if (clusterIds[i] != bestId) { clusterIds[i] = bestId; changed = true; }
        }
        if (!changed) { cout << "[KMeansSeq] Converged at iteration " << iter << "\n"; break; }

        // Update centroids
        vector<float> sumX(k, 0), sumY(k, 0), sumZ(k, 0);
        vector<int>   cnt(k, 0);
        for (int i = 0; i < n; ++i) {
            int c = clusterIds[i];
            sumX[c] += points[i].x; sumY[c] += points[i].y; sumZ[c] += points[i].z;
            cnt[c]++;
        }
        for (int c = 0; c < k; ++c) {
            if (cnt[c] > 0) { cX[c] = sumX[c]/cnt[c]; cY[c] = sumY[c]/cnt[c]; cZ[c] = sumZ[c]/cnt[c]; }
        }
    }

    // Build cluster point lists
    vector<int> clusterSize(k, 0);
    for (int i = 0; i < n; ++i) clusterSize[clusterIds[i]]++;

    vector<int> clusterStart(k, 0);
    for (int c = 1; c < k; ++c) clusterStart[c] = clusterStart[c-1] + clusterSize[c-1];

    vector<int> sortedPoints(n);
    vector<int> offset = clusterStart;
    for (int i = 0; i < n; ++i) sortedPoints[offset[clusterIds[i]]++] = i;

    // Equalize
    for (int i = 0; i < n; ++i) {
        int c     = clusterIds[i];
        int start = clusterStart[c];
        int m     = clusterSize[c];
        vector<int> neighbors(sortedPoints.begin() + start, sortedPoints.begin() + start + m);
        auto histogram = buildHistogram(neighbors);
        points[i].newIntensity = remapIntensity(histogram, points[i].intensity, m);
    }

    writeOutput("kmeans_sequential.txt");
}


#ifndef USE_CUDA
void PointCloud::runExactKNN() {
    int n = numPoints();

    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < n; ++i) {
        auto neighbors = findKNN(i, k, n, [&](int a, int b){ return squaredDist(a, b); });
        neighbors.push_back(i);
        auto histogram = buildHistogram(neighbors);
        const_cast<Point&>(points[i]).newIntensity = remapIntensity(histogram, points[i].intensity, k + 1);
    }

    writeOutput("knn.txt");
}

void PointCloud::runApproxKNN() {
    cerr << "[PointCloud] runApproxKNN not yet implemented (CPU mode).\n";
}

void PointCloud::runKMeans() {
    cerr << "[PointCloud] runKMeans not yet implemented (CPU mode).\n";
}
#endif