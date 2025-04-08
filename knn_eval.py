import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import pandas as pd

# Load data
print("Loading data...")
y = np.load('y.npy')
subjects = np.load('subject_ids.npy')
sample_weights = np.load('sample_weights.npy')

# Filter out class 2
mask = y != 2
subjects = subjects[mask]
sample_weights = sample_weights[mask]
y = y[mask]

# Load precomputed distances array
print("Loading precomputed distances...")
distances_arr = np.load('distances_arr.npy', allow_pickle=True)

# Function to perform leave-one-out cross-validation using KNN with precomputed distances
def evaluate_knn_for_subject(subject_labels, distance_matrix, k_values=[1, 3, 5, 7]):
    n_samples = len(subject_labels)
    results = {k: {'true': [], 'pred': [], 'proba': []} for k in k_values}
    
    # Perform leave-one-out cross-validation
    for i in range(n_samples):
        # Use all samples except the current one for training
        train_indices = [j for j in range(n_samples) if j != i]
        
        # Use the precomputed distance matrix
        test_distances = distance_matrix[i, train_indices]
        
        # For each k value, find the k nearest neighbors and make prediction
        for k in k_values:
            if k >= n_samples:
                # Skip if k is larger than available samples
                continue
                
            # Find k nearest neighbors
            nearest_indices = np.argsort(test_distances)[:k]
            nearest_labels = [subject_labels[train_indices[j]] for j in nearest_indices]
            
            # Make prediction (majority vote)
            prediction = np.argmax(np.bincount(nearest_labels))
            
            # Calculate probability-like score (proportion of neighbors with each class)
            class_counts = np.bincount(nearest_labels, minlength=2)
            probability = class_counts / k
            
            # Store results
            results[k]['true'].append(subject_labels[i])
            results[k]['pred'].append(prediction)
            results[k]['proba'].append(probability)
    
    return results

# Function to calculate metrics from predictions
def calculate_metrics(true_labels, pred_labels, pred_probas=None):
    metrics = {
        'accuracy': accuracy_score(true_labels, pred_labels),
        'precision': precision_score(true_labels, pred_labels, zero_division=0),
        'recall': recall_score(true_labels, pred_labels, zero_division=0),
        'f1': f1_score(true_labels, pred_labels, zero_division=0),
    }
    
    # Calculate AUC if probabilities are provided and we have both classes
    if (pred_probas is not None and 
        pred_probas.shape[1] == 2 and 
        len(np.unique(true_labels)) > 1):
        try:
            metrics['auc'] = roc_auc_score(true_labels, pred_probas[:, 1])
        except ValueError:
            # Handle case where we might not have samples of both classes
            metrics['auc'] = np.nan
            
    return metrics

# Evaluate each subject
all_results = {}
subject_metrics = {}
unique_subjects = np.sort(np.unique(subjects))
k_values = [1, 3, 5, 7, 9, 11]

print(f"Evaluating KNN with leave-one-out cross-validation for {len(unique_subjects)} subjects...")
for subject_idx, subject in enumerate(tqdm(unique_subjects)):
    # Get labels for the current subject
    subject_mask = (subjects == subject)
    subject_labels = y[subject_mask]
    
    # Skip subjects with too few samples or only one class
    if len(subject_labels) < 3 or len(np.unique(subject_labels)) < 2:
        print(f"Skipping subject {subject}: {len(subject_labels)} samples, {len(np.unique(subject_labels))} classes")
        continue
    
    # Get the precomputed distance matrix for this subject
    distance_matrix = distances_arr[subject_idx]
    
    # Evaluate KNN with different k values
    subject_results = evaluate_knn_for_subject(subject_labels, distance_matrix, k_values)
    all_results[subject] = subject_results
    
    # Calculate metrics for each k value for this subject
    subject_metrics[subject] = {}
    for k in subject_results:
        if not subject_results[k]['true']:
            continue
        
        true_labels = np.array(subject_results[k]['true'])
        pred_labels = np.array(subject_results[k]['pred'])
        pred_probas = np.array(subject_results[k]['proba'])
        
        subject_metrics[subject][k] = calculate_metrics(true_labels, pred_labels, pred_probas)

# Calculate overall metrics for each k value across all subjects
overall_metrics = {}
for k in k_values:
    all_true = []
    all_pred = []
    all_proba = []
    
    for subject in all_results:
        if k in all_results[subject]:
            all_true.extend(all_results[subject][k]['true'])
            all_pred.extend(all_results[subject][k]['pred'])
            all_proba.extend(all_results[subject][k]['proba'])
    
    if not all_true:  # Skip empty results
        continue
        
    # Convert to numpy arrays
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_proba = np.array(all_proba)
    
    # Calculate metrics
    overall_metrics[k] = calculate_metrics(all_true, all_pred, all_proba)

# Print overall results
print("\nOverall Results for each k value:")
for k in sorted(overall_metrics.keys()):
    print(f"\nk={k}:")
    for metric_name, value in overall_metrics[k].items():
        print(f"  {metric_name}: {value:.4f}")

# Create a DataFrame for subject-wise metrics
subject_df_rows = []
for subject, k_metrics in subject_metrics.items():
    for k, metrics in k_metrics.items():
        row = {'subject': subject, 'k': k}
        row.update(metrics)
        subject_df_rows.append(row)

subject_df = pd.DataFrame(subject_df_rows)

# Print top 5 subjects with best performance for k=5 (or nearest available k)
best_k = 5
if best_k not in subject_df['k'].unique():
    best_k = subject_df['k'].unique()[0]

print(f"\nTop 5 subjects with best precision (k={best_k}):")
top_subjects = subject_df[subject_df['k'] == best_k].sort_values('precision', ascending=False).head(5)
print(top_subjects[['subject', 'accuracy', 'precision', 'recall', 'f1', 'auc']])

print(f"\nBottom 5 subjects with worst precision (k={best_k}):")
bottom_subjects = subject_df[subject_df['k'] == best_k].sort_values('precision', ascending=True).head(5)
print(bottom_subjects[['subject', 'accuracy', 'precision', 'recall', 'f1', 'auc']])

# Save results
results_dict = {
    'all_results': all_results,
    'overall_metrics': overall_metrics,
    'subject_metrics': subject_metrics
}
np.save('knn_leave_one_out_results.npy', results_dict)

# Save subject metrics to CSV for easier analysis
subject_df.to_csv('subject_knn_metrics.csv', index=False)

print("\nResults saved to knn_leave_one_out_results.npy")
print("Subject metrics saved to subject_knn_metrics.csv")

# Plot overall metrics vs k
plt.figure(figsize=(10, 6))
metric_names = ['accuracy', 'precision', 'recall', 'f1']
for metric_name in metric_names:
    values = [overall_metrics[k][metric_name] for k in sorted(overall_metrics.keys()) if metric_name in overall_metrics[k]]
    plt.plot(sorted(overall_metrics.keys()), values, marker='o', label=metric_name)

plt.xlabel('k value')
plt.ylabel('Metric value')
plt.title('Overall KNN performance with different k values (Riemann distance)')
plt.legend()
plt.grid(True)
plt.savefig('knn_metrics_vs_k.pdf', format='pdf')
plt.close()

# Visualize subject-wise metrics distribution for k=best_k
plt.figure(figsize=(12, 8))
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
k_subset = subject_df[subject_df['k'] == best_k]

plt.boxplot([k_subset[metric] for metric in metrics_to_plot], labels=metrics_to_plot)
plt.title(f'Distribution of metrics across subjects for k={best_k}')
plt.ylabel('Score')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('subject_metrics_distribution.pdf', format='pdf')
plt.close()

# Plot histogram of F1 scores to see distribution
plt.figure(figsize=(10, 6))
plt.hist(k_subset['f1'], bins=20, alpha=0.7)
plt.axvline(x=k_subset['f1'].mean(), color='r', linestyle='--', label=f'Mean F1: {k_subset["f1"].mean():.2f}')
plt.xlabel('F1 Score')
plt.ylabel('Number of Subjects')
plt.title(f'Distribution of F1 Scores Across Subjects (k={best_k})')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('f1_distribution.pdf', format='pdf')
plt.close()

print("Additional plots saved: subject_metrics_distribution.pdf and f1_distribution.pdf")
