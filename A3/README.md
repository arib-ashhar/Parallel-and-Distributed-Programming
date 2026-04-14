# A3 MPI Branch-and-Bound Solver

This directory contains the MPI solution for the maximum-profit budgeted clique assignment.

The solver uses:
- branch-and-bound search
- structural bound using greedy coloring
- fractional knapsack bound
- MPI-based parallel exploration of the search frontier

## Build And Run

On the HPC setup used for this assignment:

```bash
module purge
module load compiler/gcc/9.1/mpich/3.3.1

make clean
make
```

Run a single test manually:

```bash
mpirun -np 16 ./main input_report.txt output.txt
```

Run the benchmark script:

```bash
bash benchmark.sh
```

## Final Benchmark Results

The following numbers were obtained on the provided report testcase `input_report.txt` using one node and up to 16 MPI processes.

| Processes | Execution Time (ms) | Wall Time (s) | Speedup | Efficiency | Profit |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1  | 3777 | 4.32 | 1.000 | 1.000 | 650 |
| 2  | 2070 | 2.60 | 1.825 | 0.912 | 650 |
| 4  | 1039 | 1.58 | 3.635 | 0.909 | 650 |
| 8  | 544  | 1.14 | 6.943 | 0.868 | 650 |
| 16 | 278  | 0.95 | 13.586 | 0.849 | 650 |

Notes:
- `Execution Time (ms)` is printed by the program and measures the internal solve section.
- `Wall Time (s)` is measured by the benchmark script using `/usr/bin/time`.
- All runs produce the optimal profit `650` on `input_report.txt`.

## Generated Graphs

Benchmark plots are generated from `benchmark_data.csv` and saved in `figures/`.

Available graph files:
- `figures/execution_time.svg`
- `figures/speedup.svg`
- `figures/efficiency.svg`

These graphs show:
- execution time versus number of MPI processes
- speedup relative to `np=1`
- parallel efficiency across process counts

## Report Files

This directory also contains a ready-to-use LaTeX report setup:

- `report.tex`
- `benchmark_data.csv`
- `plot_benchmarks.py`
- `figures/execution_time.svg`
- `figures/speedup.svg`
- `figures/efficiency.svg`

To regenerate the figures:

```bash
python3 plot_benchmarks.py
```

Then upload the following to Overleaf:
- `report.tex`
- `figures/`

If Overleaf does not compile SVG directly, either enable shell escape or convert the SVG files to PDF before upload.

## Implementation Highlights

The final optimized version includes:
- flattened adjacency bitsets for better cache locality
- flattened color-class storage inside the structural bound
- bucket-based ordering by profit in the structural bound
- unrolled bitset intersection checks for the report graph word width
- stronger greedy incumbent initialization before the full search
- deeper MPI frontier splitting for better parallel work distribution

## Files

- `main.cpp`: MPI solver
- `generate_testcase.cpp`: testcase generator
- `benchmark.sh`: benchmark runner
- `benchmark_data.csv`: final report benchmark data
- `plot_benchmarks.py`: script to generate graph images
- `report.tex`: LaTeX report source
