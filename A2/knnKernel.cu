#include "PointCloud.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdexcept>

using namespace std;

#define TILE_SIZE  128
#define BLOCK_SIZE 128
#define MAX_K      512


// Device helpers

__device__ float devSquaredDist(float xi, float yi, float zi, float xj, float yj, float zj) {
    float dx = xi - xj, dy = yi - yj, dz = zi - zj;
    return dx*dx + dy*dy + dz*dz;
}

__device__ void buildMaxHeap(float* heapDist, int* heapIdx, int k) {
    for (int pos = k/2 - 1; pos >= 0; --pos) {
        int cur = pos;
        while (true) {
            int left = 2*cur+1, right = 2*cur+2, largest = cur;
            if (left  < k && (heapDist[left]  > heapDist[largest] || (heapDist[left]  == heapDist[largest] && heapIdx[left]  > heapIdx[largest]))) largest = left;
            if (right < k && (heapDist[right] > heapDist[largest] || (heapDist[right] == heapDist[largest] && heapIdx[right] > heapIdx[largest]))) largest = right;
            if (largest == cur) break;
            float td = heapDist[cur]; heapDist[cur] = heapDist[largest]; heapDist[largest] = td;
            int   ti = heapIdx[cur];  heapIdx[cur]  = heapIdx[largest];  heapIdx[largest]  = ti;
            cur = largest;
        }
    }
}

__device__ void heapReplaceRoot(float* heapDist, int* heapIdx, int k, float dist, int idx) {
    heapDist[0] = dist;
    heapIdx[0]  = idx;
    int pos = 0;
    while (true) {
        int left = 2*pos+1, right = 2*pos+2, largest = pos;
        if (left  < k && (heapDist[left]  > heapDist[largest] || (heapDist[left]  == heapDist[largest] && heapIdx[left]  > heapIdx[largest]))) largest = left;
        if (right < k && (heapDist[right] > heapDist[largest] || (heapDist[right] == heapDist[largest] && heapIdx[right] > heapIdx[largest]))) largest = right;
        if (largest == pos) break;
        float td = heapDist[pos]; heapDist[pos] = heapDist[largest]; heapDist[largest] = td;
        int   ti = heapIdx[pos];  heapIdx[pos]  = heapIdx[largest];  heapIdx[largest]  = ti;
        pos = largest;
    }
}


// KNN Kernel — one thread per query point

__global__ void knnKernel(const float* __restrict__ xs, const float* __restrict__ ys,
                           const float* __restrict__ zs, const int* __restrict__ intensities,
                           int* newIntensities, int n, int k) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float xi = xs[i], yi = ys[i], zi = zs[i];

    float heapDist[MAX_K];
    int   heapIdx [MAX_K];
    int   heapSize = 0;

    __shared__ float tileX[TILE_SIZE];
    __shared__ float tileY[TILE_SIZE];
    __shared__ float tileZ[TILE_SIZE];
    __shared__ int   tileIdx[TILE_SIZE];

    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        // Cooperatively load tile — guard against out-of-bounds
        int jGlobal = t * TILE_SIZE + threadIdx.x;
        if (jGlobal < n) {
            tileX[threadIdx.x]   = xs[jGlobal];
            tileY[threadIdx.x]   = ys[jGlobal];
            tileZ[threadIdx.x]   = zs[jGlobal];
            tileIdx[threadIdx.x] = jGlobal;
        }
        __syncthreads();

        // Only process valid points in this tile
        int tileEnd = min(TILE_SIZE, n - t * TILE_SIZE);
        for (int jLocal = 0; jLocal < tileEnd; ++jLocal) {
            int jIdx = tileIdx[jLocal];
            if (jIdx == i) continue;

            float d = devSquaredDist(xi, yi, zi, tileX[jLocal], tileY[jLocal], tileZ[jLocal]);

            if (heapSize < k) {
                heapDist[heapSize] = d;
                heapIdx [heapSize] = jIdx;
                ++heapSize;
                if (heapSize == k) buildMaxHeap(heapDist, heapIdx, k);
            } else if (d < heapDist[0] || (d == heapDist[0] && jIdx < heapIdx[0])) {
                heapReplaceRoot(heapDist, heapIdx, k, d, jIdx);
            }
        }
        __syncthreads();
    }

    // Build histogram over k neighbours + point i itself
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
    if (mVal == cdfMin) {
        newIntensities[i] = intensities[i];
        return;
    }

    float remapped = (float)(cdf[intensities[i]] - cdfMin) / (float)(mVal - cdfMin) * 255.0f;
    int result = (int)(remapped + 0.5f);
    newIntensities[i] = result < 0 ? 0 : (result > 255 ? 255 : result);
}


// Host launcher

void PointCloud::runExactKNN() {
    int n = numPoints();

    vector<float> hX(n), hY(n), hZ(n);
    vector<int>   hIntensity(n), hNewIntensity(n);

    for (int i = 0; i < n; ++i) {
        hX[i]         = points[i].x;
        hY[i]         = points[i].y;
        hZ[i]         = points[i].z;
        hIntensity[i] = points[i].intensity;
    }

    float *dX, *dY, *dZ;
    int   *dIntensity, *dNewIntensity;

    cudaMalloc(&dX,            n * sizeof(float));
    cudaMalloc(&dY,            n * sizeof(float));
    cudaMalloc(&dZ,            n * sizeof(float));
    cudaMalloc(&dIntensity,    n * sizeof(int));
    cudaMalloc(&dNewIntensity, n * sizeof(int));

    cudaMemcpy(dX,         hX.data(),         n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dY,         hY.data(),         n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dZ,         hZ.data(),         n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dIntensity, hIntensity.data(), n * sizeof(int),   cudaMemcpyHostToDevice);

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    knnKernel<<<gridSize, BLOCK_SIZE>>>(dX, dY, dZ, dIntensity, dNewIntensity, n, k);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw runtime_error(string("CUDA kernel failed: ") + cudaGetErrorString(err));

    cudaMemcpy(hNewIntensity.data(), dNewIntensity, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i)
        points[i].newIntensity = hNewIntensity[i];

    cudaFree(dX); cudaFree(dY); cudaFree(dZ);
    cudaFree(dIntensity); cudaFree(dNewIntensity);

    writeOutput("knn.txt");
}
