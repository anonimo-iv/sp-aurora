#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv('comprehensive_results.csv')

# Create figure
plt.figure(figsize=(10, 6))

# Plot each implementation
for impl in df['implementation'].unique():
    impl_data = df[(df['implementation'] == impl) & (df['time_ms'].notna())]
    if not impl_data.empty:
        plt.plot(impl_data['seq_len'], impl_data['time_ms'], 
                marker='o', label=impl, linewidth=2)

plt.xlabel('Sequence Length')
plt.ylabel('Time (ms)')
plt.title('Execution Time vs Sequence Length')
plt.xscale('log', base=2)
plt.yscale('log')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('time_plot.png', dpi=150, bbox_inches='tight')
print("Plot saved to time_plot.png")