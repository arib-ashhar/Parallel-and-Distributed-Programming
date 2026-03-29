## A1

Benchmark Results

### Performance Summary

Tests conducted on **Apple M4, 10 cores** with varying workload sizes:

| Function | Speedup @ 8 threads |
|----------|---------------------|
| **totalAmountTraded()** | **5.61x** |
| **printOrderStats()** | **5.11x** |
| **updateDisplay()** | **0.99x** |

![Benchmark Results](./A1/tester/benchmark_results.png)

*Speedup vs. number of threads for the three parallel functions. Dashed line represents ideal linear speedup.*

![Detailed Overview](./A1/tester/README.md)

## A2

Benchmark results

Test Conducted on Tesla K40m GPU (compute capability 3.5, 11 GB
GDDR5) with 8 CPU cores (OpenMP). CUDA 11.0 with GCC 9.1 was used. k = 50,
T = 20.

![](./A2/results/2026-03-29_20-46.png)
![](./A2/results/2026-03-29_20-47.png)