import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data from the experiment results
data = {
    'Model': ['baseline', 'pruning', 'early pruning', 'baseline soup', 'pruning soup', 'early pruning soup', 
              'baseline soup', 'pruning soup', 'early pruning soup'],
    'Soup Type': ['none', 'none', 'none', 'uniform', 'uniform', 'uniform', 'fisher', 'fisher', 'fisher'],
    'AUC': [86.493601, 86.2672, 86.7857325, 84.8971219, 86.4941692, 84.6405774, 87.1739767, 87.5995353, 87.9105269]
}

df = pd.DataFrame(data)

# Create visualization
plt.figure(figsize=(14, 8))

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

# Create grouped bar plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Grouped bar chart
models = ['baseline', 'pruning', 'early pruning']
none_values = [86.493601, 86.2672, 86.7857325]
uniform_values = [84.8971219, 86.4941692, 84.6405774]
fisher_values = [87.1739767, 87.5995353, 87.9105269]

x = np.arange(len(models))
width = 0.25

bars1 = ax1.bar(x - width, none_values, width, label='No Soup', color='#3498db', alpha=0.8)
bars2 = ax1.bar(x, uniform_values, width, label='Uniform Soup', color='#e74c3c', alpha=0.8)
bars3 = ax1.bar(x + width, fisher_values, width, label='Fisher Soup', color='#2ecc71', alpha=0.8)

ax1.set_xlabel('Model Type')
ax1.set_ylabel('AUC Score')
ax1.set_title('AUC Performance Comparison by Model and Soup Type')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add value labels on bars
def add_value_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

add_value_labels(ax1, bars1)
add_value_labels(ax1, bars2)
add_value_labels(ax1, bars3)

# Plot 2: Heatmap
pivot_df = df.pivot(index='Model', columns='Soup Type', values='AUC')
pivot_df = pivot_df.reindex(['baseline', 'pruning', 'early pruning'])
pivot_df = pivot_df[['none', 'uniform', 'fisher']]

sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='RdYlGn', 
            cbar_kws={'label': 'AUC Score'}, ax=ax2)
ax2.set_title('AUC Performance Heatmap')
ax2.set_xlabel('Soup Type')
ax2.set_ylabel('Model Type')

plt.tight_layout()
plt.savefig('/Users/hwangsolhee/Desktop/mlpr/VAD_soup/visualizations/experiment_results_comparison.png', 
            dpi=300, bbox_inches='tight')
plt.show()

# Create a summary table
print("\n=== Experiment Results Summary ===")
print(df.to_string(index=False))

# Calculate improvements
print("\n=== Performance Analysis ===")
baseline_none = 86.493601
for _, row in df.iterrows():
    if row['Soup Type'] != 'none':
        improvement = row['AUC'] - baseline_none
        print(f"{row['Model']} with {row['Soup Type']} soup: {improvement:+.2f} AUC improvement")

# Best performers
print(f"\nBest overall performance: {df.loc[df['AUC'].idxmax(), 'Model']} with {df.loc[df['AUC'].idxmax(), 'Soup Type']} soup ({df['AUC'].max():.2f} AUC)")

# Soup type effectiveness
print("\n=== Soup Type Effectiveness ===")
for soup_type in ['uniform', 'fisher']:
    soup_data = df[df['Soup Type'] == soup_type]
    none_data = df[df['Soup Type'] == 'none']
    avg_improvement = soup_data['AUC'].mean() - none_data['AUC'].mean()
    print(f"{soup_type.title()} soup average improvement: {avg_improvement:+.2f} AUC")