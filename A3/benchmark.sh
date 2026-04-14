#!/bin/bash
set -e

echo "Building..."
make

mkdir -p benchmark_results

if [ -f input_report.txt ]; then
    TESTS=("input_report.txt")
else
    ./generate 200 500 0.4 benchmark_results/test_200.txt
    ./generate 350 800 0.4 benchmark_results/test_350.txt
    TESTS=("benchmark_results/test_200.txt" "benchmark_results/test_350.txt")
fi

PROCESSES=(1 2 4 8 16)

echo "Running benchmarks..."

for test_file in "${TESTS[@]}"; do
    test_name=$(basename "$test_file" .txt)
    echo "======================================"
    echo "Test: $test_name"
    echo "======================================"

    for np in "${PROCESSES[@]}"; do
        output_file="benchmark_results/${test_name}_np${np}.txt"
        time_file="benchmark_results/time_${test_name}_np${np}.txt"

        echo "np=$np"
        /usr/bin/time -f "%e" -o "$time_file" mpirun -np "$np" ./main "$test_file" "$output_file"
        runtime=$(cat "$time_file")
        profit=$(head -1 "$output_file")
        echo "profit=$profit time=${runtime}s"
    done
    echo ""
done
