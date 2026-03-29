#include "PointCloud.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstring>

using namespace std;

#define BLOCK_SIZE 128


// Kernel 1: Assign each point to nearest centroid
// Returns 1 in changed[] if any assignment changed (used for convergence)

__global__ void assignClustersKernel(const float* __restrict__ xs, const float* __restrict__ ys,
                                      const float* __restrict__ zs,
                                      const float* __restrict__ cX, const float* __restrict__ cY,
                                      const float* __restrict__ cZ,
                                      int* clusterIds, int* changed, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float xi = xs[i], yi = ys[i], zi = zs[i];
    float bestDist = 1e30f;
    int   bestId   = 0;

    for (int c = 0; c < k; ++c) {
        float dx = xi - cX[c], dy = yi - cY[c], dz = zi - cZ[c];
        float d  = dx*dx + dy*dy + dz*dz;
        if (d < bestDist) { bestDist = d; bestId = c; }
    }

    if (clusterIds[i] != bestId) {
        clusterIds[i] = bestId;
        *changed = 1;
    }
}


// Kernel 2: Accumulate cluster sums (partial — one block per cluster chunk)
// Uses atomicAdd to accumulate into sumX/sumY/sumZ and clusterSize

__global__ void accumulateCentroidsKernel(const float* __restrict__ xs, const float* __restrict__ ys,
                                           const float* __restrict__ zs, const int* __restrict__ clusterIds,
                                           float* sumX, float* sumY, float* sumZ,
                                           int* clusterSize, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int c = clusterIds[i];
    atomicAdd(&sumX[c],        xs[i]);
    atomicAdd(&sumY[c],        ys[i]);
    atomicAdd(&sumZ[c],        zs[i]);
    atomicAdd(&clusterSize[c], 1);
}


// Kernel 3: Update centroids = sum / size

__global__ void updateCentroidsKernel(float* cX, float* cY, float* cZ,
                                       const float* sumX, const float* sumY, const float* sumZ,
                                       const int* clusterSize, int k) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) return;
    if (clusterSize[c] == 0) return;
    cX[c] = sumX[c] / clusterSize[c];
    cY[c] = sumY[c] / clusterSize[c];
    cZ[c] = sumZ[c] / clusterSize[c];
}


// Kernel 4: Histogram equalization per point using cluster membership
// For each point i: gather all points in same cluster, build histogram,
// compute CDF, remap intensity. m = clusterSize[clusterIds[i]]

__global__ void kmeansEqualizeKernel(const int* __restrict__ intensities,
                                      const int* __restrict__ clusterIds,
                                      const int* __restrict__ sortedPoints,
                                      const int* __restrict__ clusterStart,
                                      const int* __restrict__ clusterEnd,
                                      const int* __restrict__ clusterSize,
                                      int* newIntensities, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int c     = clusterIds[i];
    int start = clusterStart[c];
    int end   = clusterEnd[c];
    int m     = clusterSize[c];

    // Build histogram over entire cluster
    int histogram[256];
    for (int v = 0; v < 256; ++v) histogram[v] = 0;
    for (int p = start; p < end; ++p)
        histogram[intensities[sortedPoints[p]]]++;

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

    if (m == cdfMin) { newIntensities[i] = intensities[i]; return; }

    float remapped = (float)(cdf[intensities[i]] - cdfMin) / (float)(m - cdfMin) * 255.0f;
    int result = (int)(remapped + 0.5f);
    newIntensities[i] = result < 0 ? 0 : (result > 255 ? 255 : result);
}


// Host launcher

void PointCloud::runKMeans() {
    int n = numPoints();

    // Flatten to SoA
    vector<float> hX(n), hY(n), hZ(n);
    vector<int>   hIntensity(n), hNewIntensity(n);
    for (int i = 0; i < n; ++i) {
        hX[i] = points[i].x; hY[i] = points[i].y; hZ[i] = points[i].z;
        hIntensity[i] = points[i].intensity;
    }

    // Init centroids = first k points
    vector<float> hCX(k), hCY(k), hCZ(k);
    for (int c = 0; c < k; ++c) {
        hCX[c] = hX[c]; hCY[c] = hY[c]; hCZ[c] = hZ[c];
    }

    // Allocate device memory
    float *dX, *dY, *dZ, *dCX, *dCY, *dCZ, *dSumX, *dSumY, *dSumZ;
    int   *dIntensity, *dNewIntensity, *dClusterIds, *dClusterSize, *dChanged;
    int   *dSortedPoints, *dClusterStart, *dClusterEnd;

    cudaMalloc(&dX,            n * sizeof(float));
    cudaMalloc(&dY,            n * sizeof(float));
    cudaMalloc(&dZ,            n * sizeof(float));
    cudaMalloc(&dCX,           k * sizeof(float));
    cudaMalloc(&dCY,           k * sizeof(float));
    cudaMalloc(&dCZ,           k * sizeof(float));
    cudaMalloc(&dSumX,         k * sizeof(float));
    cudaMalloc(&dSumY,         k * sizeof(float));
    cudaMalloc(&dSumZ,         k * sizeof(float));
    cudaMalloc(&dIntensity,    n * sizeof(int));
    cudaMalloc(&dNewIntensity, n * sizeof(int));
    cudaMalloc(&dClusterIds,   n * sizeof(int));
    cudaMalloc(&dClusterSize,  k * sizeof(int));
    cudaMalloc(&dChanged,          sizeof(int));
    cudaMalloc(&dSortedPoints, n * sizeof(int));
    cudaMalloc(&dClusterStart, k * sizeof(int));
    cudaMalloc(&dClusterEnd,   k * sizeof(int));

    cudaMemcpy(dX,         hX.data(),         n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dY,         hY.data(),         n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dZ,         hZ.data(),         n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dCX,        hCX.data(),        k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dCY,        hCY.data(),        k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dCZ,        hCZ.data(),        k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dIntensity, hIntensity.data(), n * sizeof(int),   cudaMemcpyHostToDevice);

    // Init clusterIds to 0
    cudaMemset(dClusterIds, 0, n * sizeof(int));

    int gridN = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int gridK = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // --------------- K-Means iterations ---------------
    for (int iter = 0; iter < maxIter; ++iter) {
        // Reset changed flag
        cudaMemset(dChanged, 0, sizeof(int));

        // Step 1: assign
        assignClustersKernel<<<gridN, BLOCK_SIZE>>>(dX, dY, dZ, dCX, dCY, dCZ, dClusterIds, dChanged, n, k);
        cudaDeviceSynchronize();

        // Check convergence
        int hChanged = 0;
        cudaMemcpy(&hChanged, dChanged, sizeof(int), cudaMemcpyDeviceToHost);
        if (hChanged == 0) {
            cout << "[KMeans] Converged at iteration " << iter << "\n";
            break;
        }

        // Step 2: reset accumulators
        cudaMemset(dSumX,        0, k * sizeof(float));
        cudaMemset(dSumY,        0, k * sizeof(float));
        cudaMemset(dSumZ,        0, k * sizeof(float));
        cudaMemset(dClusterSize, 0, k * sizeof(int));

        // Step 3: accumulate
        accumulateCentroidsKernel<<<gridN, BLOCK_SIZE>>>(dX, dY, dZ, dClusterIds, dSumX, dSumY, dSumZ, dClusterSize, n);
        cudaDeviceSynchronize();

        // Step 4: update centroids
        updateCentroidsKernel<<<gridK, BLOCK_SIZE>>>(dCX, dCY, dCZ, dSumX, dSumY, dSumZ, dClusterSize, k);
        cudaDeviceSynchronize();
    }

    // --------------- Build sorted cluster arrays on host ---------------
    // Copy final clusterIds and clusterSize back
    vector<int> hClusterIds(n), hClusterSize(k, 0);
    cudaMemcpy(hClusterIds.data(),  dClusterIds,  n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hClusterSize.data(), dClusterSize, k * sizeof(int), cudaMemcpyDeviceToHost);

    // Counting sort: sort point indices by cluster id
    vector<int> hClusterStart(k, 0), hClusterEnd(k, 0);
    vector<int> count(k, 0);
    for (int i = 0; i < n; ++i) count[hClusterIds[i]]++;
    hClusterStart[0] = 0;
    for (int c = 1; c < k; ++c) hClusterStart[c] = hClusterStart[c-1] + count[c-1];
    for (int c = 0; c < k; ++c) hClusterEnd[c]   = hClusterStart[c] + count[c];

    vector<int> hSortedPoints(n);
    vector<int> offset = hClusterStart;
    for (int i = 0; i < n; ++i) {
        hSortedPoints[offset[hClusterIds[i]]++] = i;
    }

    // Copy sorted structures to device
    cudaMemcpy(dSortedPoints, hSortedPoints.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dClusterStart, hClusterStart.data(), k * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dClusterEnd,   hClusterEnd.data(),   k * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dClusterSize,  hClusterSize.data(),  k * sizeof(int), cudaMemcpyHostToDevice);

    // --------------- Equalization kernel ---------------
    kmeansEqualizeKernel<<<gridN, BLOCK_SIZE>>>(dIntensity, dClusterIds, dSortedPoints,
                                                 dClusterStart, dClusterEnd, dClusterSize,
                                                 dNewIntensity, n);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw runtime_error(string("KMeans CUDA kernel failed: ") + cudaGetErrorString(err));

    cudaMemcpy(hNewIntensity.data(), dNewIntensity, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i)
        points[i].newIntensity = hNewIntensity[i];

    // Free
    cudaFree(dX); cudaFree(dY); cudaFree(dZ);
    cudaFree(dCX); cudaFree(dCY); cudaFree(dCZ);
    cudaFree(dSumX); cudaFree(dSumY); cudaFree(dSumZ);
    cudaFree(dIntensity); cudaFree(dNewIntensity);
    cudaFree(dClusterIds); cudaFree(dClusterSize); cudaFree(dChanged);
    cudaFree(dSortedPoints); cudaFree(dClusterStart); cudaFree(dClusterEnd);

    writeOutput("kmeans.txt");
}
