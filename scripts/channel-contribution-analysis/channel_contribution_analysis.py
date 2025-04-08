import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
import pandas as pd
import os
from scipy.stats import pearsonr
from config import PICK_CHANNELS
from tqdm import tqdm

def compute_csp_patterns_for_subject(X_subject, y_subject):
    """Compute CSP patterns for a single subject"""
    # Compute covariances
    cov = Covariances(estimator="oas")
    X_cov = cov.transform(X_subject)
    
    # Fit CSP
    csp = CSP(nfilter=4, metric='riemann', log=True)
    csp.fit(X_cov, y_subject)
    
    # Get patterns (spatial filters)
    patterns = csp.patterns_
    # for ind, pattern in enumerate(patterns):
    patterns = np.abs(patterns) 
    patterns = (patterns - np.min(patterns)) / (np.max(patterns) - np.min(patterns))
    
    return patterns

# Load data
print("Loading data...")
X = np.load('X.npy')
y = np.load('y.npy')
subjects = np.load('subject_ids.npy')

# Filter out class 2
mask = y != 2
X = X[mask]
subjects = subjects[mask]
y = y[mask]

# Load KNN results
print("Loading KNN evaluation results...")
try:
    subject_metrics_df = pd.read_csv('subject_knn_metrics.csv')
except FileNotFoundError:
    print("Error: subject_knn_metrics.csv not found. Run knn_eval.py first.")
    exit(1)

# Get precision scores for k=5 (or another k if 5 is not available)
k_value = 5
if k_value not in subject_metrics_df['k'].unique():
    k_value = subject_metrics_df['k'].unique()[0]

precision_scores = {}
for _, row in subject_metrics_df[subject_metrics_df['k'] == k_value].iterrows():
    precision_scores[row['subject']] = row['precision']

# Create a dataframe to store all channel powers and metrics
columns = ['subject', 'channel', 'filter', 'power', 'precision']
data_rows = []

# Process each subject
print(f"Analyzing CSP patterns and correlating with precision scores (k={k_value})...")
unique_subjects = np.sort(np.unique(subjects))

for subject in tqdm(unique_subjects):
    # Skip if no precision score available
    if subject not in precision_scores:
        continue
    
    # Get subject data
    subject_mask = subjects == subject
    X_subject = X[subject_mask]
    y_subject = y[subject_mask]
    
    # Compute CSP patterns

    patterns = compute_csp_patterns_for_subject(X_subject, y_subject)
    
    # For each filter and each channel, store the power
    for filter_idx in range(patterns.shape[0]):
        for channel_idx, channel in enumerate(PICK_CHANNELS):
            if channel_idx < patterns.shape[1]:  # Make sure the channel exists in patterns
                # Store raw power value (will apply absolute value before normalization)
                power = patterns[filter_idx, channel_idx]
                # power = power - np.min(power) / (np.max(power) - np.min(power))
                
                # Store in dataframe
                data_rows.append({
                    'subject': subject,
                    'channel': channel,
                    'filter': f"Filter {filter_idx+1}",
                    'power': power,
                    'precision': precision_scores[subject]
                })

# Create dataframe from collected data
df = pd.DataFrame(data_rows)

# # Normalize power values for each filter separately to range [0,1]
# print("Normalizing power values for each filter to range [0,1]...")
# normalized_df = df.copy()

# for filter_name in df['filter'].unique():
#     filter_mask = df['filter'] == filter_name
#     min_power = df.loc[filter_mask, 'power'].min()
#     max_power = df.loc[filter_mask, 'power'].max()
    
#     # Apply min-max normalization
#     if max_power > min_power:  # Avoid division by zero
#         normalized_df.loc[filter_mask, 'power'] = (df.loc[filter_mask, 'power'] - min_power) / (max_power - min_power)
#     else:
#         normalized_df.loc[filter_mask, 'power'] = 0.5  # Default value if all powers are the same

# # Use the normalized dataframe for all subsequent operations
# df = normalized_df

# Calculate correlations for each channel and filter
correlation_results = []
for channel in PICK_CHANNELS:
    for filter_name in df['filter'].unique():
        channel_filter_data = df[(df['channel'] == channel) & (df['filter'] == filter_name)]
        
        # Only calculate if we have enough data
        if len(channel_filter_data) > 5:
            x = channel_filter_data['power'].values
            y = channel_filter_data['precision'].values
            
            try:
                corr, p_val = pearsonr(x, y)
                correlation_results.append({
                    'channel': channel,
                    'filter': filter_name,
                    'correlation': corr,
                    'p_value': p_val,
                    'abs_corr': abs(corr)
                })
            except:
                pass

# Create correlation dataframe
corr_df = pd.DataFrame(correlation_results)

# Find top channel-filter combinations by absolute correlation
top_combinations = corr_df.sort_values('abs_corr', ascending=False).head(10)
print("\nTop 10 channel-filter combinations by correlation strength:")
for i, row in top_combinations.iterrows():
    print(f"{i+1}. {row['channel']} - {row['filter']}: r={row['correlation']:.3f}, p={row['p_value']:.4f}")

# Create a directory for the plots
os.makedirs("channel_correlation_plots", exist_ok=True)

# Create separate plot for each filter with 18 subplots (one per channel)
for filter_name in df['filter'].unique():
    print(f"Creating plot for {filter_name} with 18 channel subplots...")
    
    # Get data for this filter
    filter_data = df[df['filter'] == filter_name]
    
    # Create a figure with a grid of subplots (6x3 grid)
    fig, axes = plt.subplots(6, 3, figsize=(15, 20))
    axes = axes.flatten()  # Flatten for easy indexing
    
    # Sort channels by correlation strength for this filter
    filter_corrs = corr_df[corr_df['filter'] == filter_name].sort_values('abs_corr', ascending=False)
    channel_corr_dict = {row['channel']: row['correlation'] for _, row in filter_corrs.iterrows()}
    
    # Plot each channel in its own subplot
    for i, channel in enumerate(PICK_CHANNELS):
        ax = axes[i]
        channel_data = filter_data[filter_data['channel'] == channel]
        
        # Skip if not enough data
        if len(channel_data) < 5:
            ax.text(0.5, 0.5, f"{channel}: insufficient data", 
                    ha='center', va='center', transform=ax.transAxes)
            continue
            
        # Get correlation value and p-value
        corr_row = corr_df[(corr_df['channel'] == channel) & (corr_df['filter'] == filter_name)]
        if len(corr_row) > 0:
            corr = corr_row['correlation'].values[0]
            p_val = corr_row['p_value'].values[0]
            
            # Scatter plot
            x = channel_data['power'].values
            y = channel_data['precision'].values
            
            # Determine color based on correlation strength (red for positive, blue for negative)
            color = 'red' if corr > 0 else 'blue'
            alpha = min(1.0, 0.3 + abs(corr) * 0.7)  # Higher correlation = more opaque
            
            ax.scatter(x, y, alpha=0.6, color=color)
            
            # Regression line
            if len(x) > 1:  # Need at least 2 points for regression
                m, b = np.polyfit(x, y, 1)
                x_line = np.linspace(0, 1, 100)  # Use full normalized range
                y_line = m * x_line + b
                ax.plot(x_line, y_line, color=color, linewidth=2)
            
            # Add correlation coefficient and p-value
            sig_stars = ''
            if p_val < 0.001:
                sig_stars = '***'
            elif p_val < 0.01:
                sig_stars = '**'
            elif p_val < 0.05:
                sig_stars = '*'
                
            ax.set_title(f"{channel}: r={corr:.3f}{sig_stars}", fontsize=10)
        else:
            ax.text(0.5, 0.5, f"{channel}: no correlation data", 
                    ha='center', va='center', transform=ax.transAxes)
        
        # Set axes labels
        if i >= 15:  # Only add x-labels to bottom row
            ax.set_xlabel("Normalized Power")
        if i % 3 == 0:  # Only add y-labels to leftmost column
            ax.set_ylabel("Precision")
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add title to figure
    fig.suptitle(f"{filter_name}: Absolute Channel Power vs. Precision (k={k_value})", 
                 fontsize=16, y=0.995)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.98)
    
    # Save figure
    plt.savefig(f"channel_correlation_plots/{filter_name.replace(' ', '_')}_all_channels.pdf", 
                format='pdf', bbox_inches='tight')
    plt.savefig(f"channel_correlation_plots/{filter_name.replace(' ', '_')}_all_channels.png", 
                format='png', dpi=300, bbox_inches='tight')
    plt.close()

# Create a heatmap showing correlation strength for all channels and filters
print("Creating correlation heatmap...")

# Pivot the correlation dataframe to create a channel x filter matrix
pivot_df = corr_df.pivot(index='channel', columns='filter', values='correlation')

# Sort channels by mean absolute correlation
channel_order = corr_df.groupby('channel')['abs_corr'].mean().sort_values(ascending=False).index.tolist()
pivot_df = pivot_df.reindex(channel_order)

plt.figure(figsize=(10, 12))
sns.heatmap(pivot_df, annot=True, cmap='coolwarm', center=0, 
            vmin=-1, vmax=1, fmt='.2f', linewidths=.5)
plt.title(f"Correlation Between Channel Power and Precision Score (k={k_value})")
plt.tight_layout()
plt.savefig("channel_filter_correlation_heatmap.pdf", format='pdf')
plt.savefig("channel_filter_correlation_heatmap.png", format='png', dpi=300)
plt.close()

print("Analysis complete. Results saved to:")
print("1. Individual filter plots with all channels: channel_correlation_plots/")
print("2. Correlation heatmap: channel_filter_correlation_heatmap.pdf/png") 