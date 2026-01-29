import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read benchmark results
df = pd.read_csv('benchmark_results.csv')

# Get unique functions and sizes
functions = df['Function'].unique()
sizes = df['Size'].unique()

# Create figure with subplots
fig, axes = plt.subplots(len(functions), 2, figsize=(14, 5*len(functions)))

for idx, func in enumerate(functions):
    func_data = df[df['Function'] == func]
    
    # Plot 1: Speedup vs Threads
    ax1 = axes[idx, 0] if len(functions) > 1 else axes[0]
    for size in sizes:
        size_data = func_data[func_data['Size'] == size]
        ax1.plot(size_data['Threads'], size_data['Speedup'], 
                marker='o', label=f'{size:,} orders')
    
    # Add ideal speedup line
    max_threads = df['Threads'].max()
    ax1.plot([1, max_threads], [1, max_threads], 
            'k--', alpha=0.3, label='Ideal')
    
    ax1.set_xlabel('Number of Threads')
    ax1.set_ylabel('Speedup')
    ax1.set_title(f'{func} - Speedup vs Threads')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Execution Time vs Threads
    ax2 = axes[idx, 1] if len(functions) > 1 else axes[1]
    for size in sizes:
        size_data = func_data[func_data['Size'] == size]
        ax2.plot(size_data['Threads'], size_data['Time_ms'], 
                marker='o', label=f'{size:,} orders')
    
    ax2.set_xlabel('Number of Threads')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title(f'{func} - Execution Time vs Threads')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
print("Graph saved as benchmark_results.png")

# Print summary statistics
print("\n=== Performance Summary ===")
for func in functions:
    func_data = df[df['Function'] == func]
    max_size_data = func_data[func_data['Size'] == func_data['Size'].max()]
    max_threads_data = max_size_data[max_size_data['Threads'] == max_size_data['Threads'].max()]
    
    speedup = max_threads_data['Speedup'].values[0]
    print(f"{func}: {speedup:.2f}x speedup with {max_threads_data['Threads'].values[0]} threads")