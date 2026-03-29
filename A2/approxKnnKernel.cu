#include "PointCloud.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

using namespace std;

#define BLOCK_SIZE  128
#define MAX_K       512
#define MAX_RADIUS  4       // max cell search radius before giving up


// Device helpers

__device__ float approxSquaredDist(float xi, float yi, float zi, float xj, float yj, float zj) {
    float dx = xi - xj, dy = yi - yj, dz = zi - zj;
    return dx*dx + dy*dy + dz*dz;
}

__device__ void approxBuildMaxHeap(float* heapDist, int* heapIdx, int k) {
    for (int pos = k/2 - 1; pos >= 0; --pos) {
        int cur = pos;
        while (true) {
            int left = 2*cur+1, right = 2*cur+2, largest = cur;
            if (left  < k && heapDist[left]  > heapDist[largest]) largest = left;
            if (right < k && heapDist[right] > heapDist[largest]) largest = right;
            if (largest == cur) break;
            float td = heapDist[cur]; heapDist[cur] = heapDist[largest]; heapDist[largest] = td;
            int   ti = heapIdx[cur];  heapIdx[cur]  = heapIdx[largest];  heapIdx[largest]  = ti;
            cur = largest;
        }
    }
}

__device__ void approxHeapReplaceRoot(float* heapDist, int* heapIdx, int k, float dist, int idx) {
    heapDist[0] = dist;
    heapIdx[0]  = idx;
    int pos = 0;
    while (true) {
        int left = 2*pos+1, right = 2*pos+2, largest = pos;
        if (left  < k && heapDist[left]  > heapDist[largest]) largest = left;
        if (right < k && heapDist[right] > heapDist[largest]) largest = right;
        if (largest == pos) break;
        float td = heapDist[pos]; heapDist[pos] = heapDist[largest]; heapDist[largest] = td;
        int   ti = heapIdx[pos];  heapIdx[pos]  = heapIdx[largest];  heapIdx[largest]  = ti;
        pos = largest;
    }
}


// Kernel: assign each point to a voxel cell and compute cell key
// cellKey = cx + cy * gridDimX + cz * gridDimX * gridDimY

__global__ void assignCellsKernel(const float* xs, const float* ys, const float* zs,
                                   int* cellKeys, int* pointOrder,
                                   float minX, float minY, float minZ,
                                   float voxelSize, int gridDimX, int gridDimY, int gridDimZ,
                                   int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int cx = min((int)((xs[i] - minX) / voxelSize), gridDimX - 1);
    int cy = min((int)((ys[i] - minY) / voxelSize), gridDimY - 1);
    int cz = min((int)((zs[i] - minZ) / voxelSize), gridDimZ - 1);

    cellKeys[i]   = cx + cy * gridDimX + cz * gridDimX * gridDimY;
    pointOrder[i] = i;
}


// Kernel: build cellStart and cellEnd arrays from sorted cellKeys
// cellStart[c] = first index in sortedPoints[] that belongs to cell c
// cellEnd[c]   = one past last index

__global__ void buildCellArrayKernel(const int* sortedCellKeys, int* cellStart, int* cellEnd, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int cell = sortedCellKeys[i];
    if (i == 0 || sortedCellKeys[i-1] != cell)
        cellStart[cell] = i;
    if (i == n-1 || sortedCellKeys[i+1] != cell)
        cellEnd[cell] = i + 1;
}


// Main approximate KNN kernel
// For each query point i:
//   1. Find its voxel cell
//   2. Search cells in increasing radius shells until k candidates found
//   3. Run heap KNN over those candidates
//   4. Build histogram, CDF, remap

__global__ void approxKnnKernel(const float* __restrict__ xs, const float* __restrict__ ys,
                                  const float* __restrict__ zs, const int* __restrict__ intensities,
                                  const int* __restrict__ sortedPoints,
                                  const int* __restrict__ cellStart, const int* __restrict__ cellEnd,
                                  int* newIntensities,
                                  float minX, float minY, float minZ,
                                  float voxelSize, int gridDimX, int gridDimY, int gridDimZ,
                                  int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float xi = xs[i], yi = ys[i], zi = zs[i];

    int cx = min((int)((xi - minX) / voxelSize), gridDimX - 1);
    int cy = min((int)((yi - minY) / voxelSize), gridDimY - 1);
    int cz = min((int)((zi - minZ) / voxelSize), gridDimZ - 1);

    float heapDist[MAX_K];
    int   heapIdx [MAX_K];
    int   heapSize = 0;

    // Expand search radius until we have k candidates
    for (int radius = 1; radius <= MAX_RADIUS; ++radius) {
        int rLo = radius - 1;

        for (int dz = -radius; dz <= radius; ++dz) {
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    // Only process the shell at this radius (skip inner shells)
                    if (abs(dx) < rLo && abs(dy) < rLo && abs(dz) < rLo) continue;

                    int nx = cx + dx, ny = cy + dy, nz = cz + dz;
                    if (nx < 0 || nx >= gridDimX) continue;
                    if (ny < 0 || ny >= gridDimY) continue;
                    if (nz < 0 || nz >= gridDimZ) continue;

                    int cellKey = nx + ny * gridDimX + nz * gridDimX * gridDimY;
                    int start   = cellStart[cellKey];
                    int end     = cellEnd[cellKey];
                    if (start == -1) continue;

                    for (int p = start; p < end; ++p) {
                        int j = sortedPoints[p];
                        if (j == i) continue;

                        float d = approxSquaredDist(xi, yi, zi, xs[j], ys[j], zs[j]);

                        if (heapSize < k) {
                            heapDist[heapSize] = d;
                            heapIdx [heapSize] = j;
                            ++heapSize;
                            if (heapSize == k) approxBuildMaxHeap(heapDist, heapIdx, k);
                        } else if (d < heapDist[0]) {
                            approxHeapReplaceRoot(heapDist, heapIdx, k, d, j);
                        }
                    }
                }
            }
        }
        if (heapSize >= k) break;
    }

    // Build histogram over heapSize neighbours + point i itself
    int histogram[256];
    for (int v = 0; v < 256; ++v) histogram[v] = 0;
    histogram[intensities[i]]++;
    for (int m = 0; m < heapSize; ++m)
        histogram[intensities[heapIdx[m]]]++;

    // CDF
    int cdf[256];
    cdf[0] = histogram[0];
    for (int v = 1; v < 256; ++v)
        cdf[v] = cdf[v-1] + histogram[v];

    // cdfMin
    int cdfMin = 0;
    for (int v = 0; v < 256; ++v) {
        if (cdf[v] > 0) { cdfMin = cdf[v]; break; }
    }

    int mVal = k + 1;
    if (mVal == cdfMin) { newIntensities[i] = intensities[i]; return; }

    float remapped = (float)(cdf[intensities[i]] - cdfMin) / (float)(mVal - cdfMin) * 255.0f;
    int result = (int)(remapped + 0.5f);
    newIntensities[i] = result < 0 ? 0 : (result > 255 ? 255 : result);
}


// Host-side counting sort — sorts pointOrder by cellKey (GPU-friendly, O(N))

static void countingSortByCellKey(vector<int>& pointOrder, vector<int>& cellKeys,
                                   vector<int>& sortedCellKeys, int totalCells) {
    int n = pointOrder.size();
    vector<int> count(totalCells, 0);
    for (int i = 0; i < n; ++i) count[cellKeys[i]]++;

    vector<int> prefix(totalCells, 0);
    for (int c = 1; c < totalCells; ++c) prefix[c] = prefix[c-1] + count[c-1];

    vector<int> sortedOrder(n);
    sortedCellKeys.resize(n);
    vector<int> offset = prefix;
    for (int i = 0; i < n; ++i) {
        int c = cellKeys[i];
        sortedOrder[offset[c]]    = pointOrder[i];
        sortedCellKeys[offset[c]] = c;
        offset[c]++;
    }
    pointOrder = sortedOrder;
}


// Host launcher

void PointCloud::runApproxKNN() {
    int n = numPoints();

    // Flatten to SoA
    vector<float> hX(n), hY(n), hZ(n);
    vector<int>   hIntensity(n), hNewIntensity(n);
    for (int i = 0; i < n; ++i) {
        hX[i] = points[i].x; hY[i] = points[i].y; hZ[i] = points[i].z;
        hIntensity[i] = points[i].intensity;
    }

    // Bounding box
    float minX = *min_element(hX.begin(), hX.end());
    float minY = *min_element(hY.begin(), hY.end());
    float minZ = *min_element(hZ.begin(), hZ.end());
    float maxX = *max_element(hX.begin(), hX.end());
    float maxY = *max_element(hY.begin(), hY.end());
    float maxZ = *max_element(hZ.begin(), hZ.end());

    // Voxel size: target ~k points per cell on average
    // volume / n * k gives volume per cell → cube root gives cell side
    float vol      = (maxX-minX) * (maxY-minY) * (maxZ-minZ);
    float voxelSize = cbrt(vol / n * k) * 1.5f;   // 1.5x slack for better coverage
    voxelSize = max(voxelSize, 1e-4f);

    int gridDimX = max(1, (int)ceil((maxX - minX) / voxelSize));
    int gridDimY = max(1, (int)ceil((maxY - minY) / voxelSize));
    int gridDimZ = max(1, (int)ceil((maxZ - minZ) / voxelSize));
    int totalCells = gridDimX * gridDimY * gridDimZ;

    cout << "[ApproxKNN] voxelSize=" << voxelSize
         << " grid=" << gridDimX << "x" << gridDimY << "x" << gridDimZ
         << " totalCells=" << totalCells << "\n";

    // Assign points to cells and sort by cell key (host counting sort)
    vector<int> pointOrder(n), cellKeys(n);
    for (int i = 0; i < n; ++i) {
        int cx = min((int)((hX[i] - minX) / voxelSize), gridDimX - 1);
        int cy = min((int)((hY[i] - minY) / voxelSize), gridDimY - 1);
        int cz = min((int)((hZ[i] - minZ) / voxelSize), gridDimZ - 1);
        cellKeys[i]   = cx + cy * gridDimX + cz * gridDimX * gridDimY;
        pointOrder[i] = i;
    }

    vector<int> sortedCellKeys;
    countingSortByCellKey(pointOrder, cellKeys, sortedCellKeys, totalCells);

    // Build cellStart / cellEnd on host
    vector<int> hCellStart(totalCells, -1), hCellEnd(totalCells, -1);
    for (int i = 0; i < n; ++i) {
        int c = sortedCellKeys[i];
        if (hCellStart[c] == -1) hCellStart[c] = i;
        hCellEnd[c] = i + 1;
    }

    // Allocate and copy to device
    float *dX, *dY, *dZ;
    int   *dIntensity, *dNewIntensity, *dSortedPoints, *dCellStart, *dCellEnd;

    cudaMalloc(&dX,            n * sizeof(float));
    cudaMalloc(&dY,            n * sizeof(float));
    cudaMalloc(&dZ,            n * sizeof(float));
    cudaMalloc(&dIntensity,    n * sizeof(int));
    cudaMalloc(&dNewIntensity, n * sizeof(int));
    cudaMalloc(&dSortedPoints, n * sizeof(int));
    cudaMalloc(&dCellStart,    totalCells * sizeof(int));
    cudaMalloc(&dCellEnd,      totalCells * sizeof(int));

    cudaMemcpy(dX,            hX.data(),            n * sizeof(float),          cudaMemcpyHostToDevice);
    cudaMemcpy(dY,            hY.data(),             n * sizeof(float),          cudaMemcpyHostToDevice);
    cudaMemcpy(dZ,            hZ.data(),             n * sizeof(float),          cudaMemcpyHostToDevice);
    cudaMemcpy(dIntensity,    hIntensity.data(),     n * sizeof(int),            cudaMemcpyHostToDevice);
    cudaMemcpy(dSortedPoints, pointOrder.data(),     n * sizeof(int),            cudaMemcpyHostToDevice);
    cudaMemcpy(dCellStart,    hCellStart.data(),     totalCells * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(dCellEnd,      hCellEnd.data(),       totalCells * sizeof(int),   cudaMemcpyHostToDevice);

    // Launch kernel
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    approxKnnKernel<<<gridSize, BLOCK_SIZE>>>(
        dX, dY, dZ, dIntensity,
        dSortedPoints, dCellStart, dCellEnd,
        dNewIntensity,
        minX, minY, minZ,
        voxelSize, gridDimX, gridDimY, gridDimZ,
        n, k);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw runtime_error(string("ApproxKNN CUDA kernel failed: ") + cudaGetErrorString(err));

    cudaMemcpy(hNewIntensity.data(), dNewIntensity, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i)
        points[i].newIntensity = hNewIntensity[i];

    cudaFree(dX); cudaFree(dY); cudaFree(dZ);
    cudaFree(dIntensity); cudaFree(dNewIntensity);
    cudaFree(dSortedPoints); cudaFree(dCellStart); cudaFree(dCellEnd);

    writeOutput("approx_knn.txt");
}
